"""
TC-Pade model-level acceleration for ComfyUI.

Official reference:
- https://github.com/Alibaba-VELLDEPTH/TC_Pade
- TC-Pade: Trajectory-Consistent Pade Approximation for Diffusion Acceleration

This adapter keeps TC-Pade as a model-level method: it patches the diffusion
model forward path and predicts whole-transformer residuals inside the model.
"""
from unittest.mock import patch

import torch
from torch import Tensor

from comfy.ldm.flux.layers import timestep_embedding

from .pade_utils import ResidualPredictor


TCPADE_MODEL_TYPES = ["flux", "wan2.1"]


def _init_state(self, opts):
    num_steps = int(opts.get("num_steps", 50))
    if not hasattr(self, "_tcpade_predictor") or getattr(self, "_tcpade_num_steps", None) != num_steps:
        self._tcpade_predictor = ResidualPredictor(
            order=int(opts.get("predictor_order", 3)),
            history_size=int(opts.get("history_size", 6)),
            n_threshold=float(opts.get("n_threshold", 1.4)),
            cache_device=opts.get("cache_device", "cuda"),
        )
        self._tcpade_predictor.total_timesteps = num_steps
        self._tcpade_num_steps = num_steps


def _maybe_predict(self, hidden_states, opts):
    predictor = self._tcpade_predictor
    step = int(opts.get("current_step", 0))
    num_steps = int(opts.get("num_steps", 50))
    start_step = int(opts.get("start_step", 4))
    end_step = int(opts.get("end_step", -1))
    if end_step < 0:
        end_step = num_steps - 1
    interval = max(int(opts.get("interval", 8)), 1)

    use_predict = start_step < step < end_step and step % interval != 0
    if use_predict and predictor.curvetest():
        predicted = predictor.predict_output(hidden_states, step)
        if predicted is not None:
            print(f"[TC-Pade] predicted residual at step={step}")
            return predicted, True
    return hidden_states, False


def tcpade_flux_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
    **kwargs,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    opts = transformer_options.get("tcpade_options", {})
    _init_state(self, opts)

    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    hidden_in = img.clone()
    img, predicted = _maybe_predict(self, img, opts)

    if not predicted:
        extra_kwargs = {"attn_mask": attn_mask} if attn_mask is not None else {}
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], **extra_kwargs
                    )
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options},
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_kwargs)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i) and control_i[i] is not None:
                    img += control_i[i]

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], **extra_kwargs)
                    return out
                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options},
                )
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, **extra_kwargs)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o) and control_o[i] is not None:
                    img[:, txt.shape[1]:, ...] += control_o[i]

        img = img[:, txt.shape[1]:, ...]

    self._tcpade_predictor.update_history(hidden_in, img)
    return self.final_layer(img, vec)


def tcpade_wan_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    from comfy.ldm.wan.model import sinusoidal_embedding_1d

    patches_replace = transformer_options.get("patches_replace", {})
    opts = transformer_options.get("tcpade_options", {})
    _init_state(self, opts)

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context = self.text_embedding(context)
    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    hidden_in = x.clone()
    x, predicted = _maybe_predict(self, x, opts)

    if not predicted:
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"], context=args["txt"], e=args["vec"],
                        freqs=args["pe"], context_img_len=context_img_len
                    )
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": x, "txt": context, "vec": e0, "pe": freqs},
                    {"original_block": block_wrap, "transformer_options": transformer_options},
                )
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

    self._tcpade_predictor.update_history(hidden_in, x)
    x = self.head(x, e)
    return self.unpatchify(x, grid_sizes)


class TCPade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (TCPADE_MODEL_TYPES, {"default": "wan2.1"}),
                "start_step": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "interval": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "n_threshold": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 5.0, "step": 0.1}),
                "predictor_order": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1}),
                "history_size": ("INT", {"default": 6, "min": 3, "max": 12, "step": 1}),
                "cache_device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_tcpade"
    CATEGORY = "TC-Pade"
    TITLE = "TC-Pade"

    def apply_tcpade(
        self,
        model,
        model_type: str,
        start_step: int,
        end_step: int,
        interval: int,
        n_threshold: float,
        predictor_order: int,
        history_size: int,
        cache_device: str,
    ):
        new_model = model.clone()
        if "transformer_options" not in new_model.model_options:
            new_model.model_options["transformer_options"] = {}

        new_model.model_options["transformer_options"]["tcpade_options"] = {
            "start_step": start_step,
            "end_step": end_step,
            "interval": interval,
            "n_threshold": n_threshold,
            "predictor_order": predictor_order,
            "history_size": history_size,
            "cache_device": cache_device,
        }

        diffusion_model = new_model.get_model_object("diffusion_model")
        for attr in ("_tcpade_predictor", "_tcpade_num_steps"):
            if hasattr(diffusion_model, attr):
                delattr(diffusion_model, attr)

        if model_type == "flux":
            forward_fn = tcpade_flux_forward
            forward_attr = "forward_orig"
        elif model_type == "wan2.1":
            forward_fn = tcpade_wan_forward
            forward_attr = "forward_orig"
        else:
            raise ValueError(f"Unsupported TC-Pade model type: {model_type}")

        context = patch.multiple(
            diffusion_model,
            **{forward_attr: forward_fn.__get__(diffusion_model, diffusion_model.__class__)},
        )

        def unet_wrapper_function(model_function, kwargs):
            input_tensor = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]

            sigmas = c["transformer_options"].get("sample_sigmas")
            current_step = 0
            num_steps = 1
            if sigmas is not None:
                num_steps = max(len(sigmas) - 1, 1)
                matched = (sigmas == timestep[0]).nonzero()
                if len(matched) > 0:
                    current_step = int(matched.item())
                else:
                    for i in range(len(sigmas) - 1):
                        if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                            current_step = i
                            break

                if current_step == 0:
                    for attr in ("_tcpade_predictor", "_tcpade_num_steps"):
                        if hasattr(diffusion_model, attr):
                            delattr(diffusion_model, attr)

            opts = c["transformer_options"].setdefault("tcpade_options", {})
            opts.update({"current_step": current_step, "num_steps": num_steps})

            with context:
                return model_function(input_tensor, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)


def apply_tcpade(
    model,
    model_type: str,
    start_step: int,
    end_step: int,
    interval: int,
    n_threshold: float,
    predictor_order: int,
    history_size: int,
    cache_device: str,
):
    node = TCPade()
    return node.apply_tcpade(
        model,
        model_type,
        start_step,
        end_step,
        interval,
        n_threshold,
        predictor_order,
        history_size,
        cache_device,
    )[0]
