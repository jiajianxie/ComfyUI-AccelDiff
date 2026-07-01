"""
SeaCache model acceleration for ComfyUI.
Based on: https://arxiv.org/abs/2602.18993 (CVPR 2026 Oral)

Spectral-Evolution-Aware Cache: uses Wiener filtering to improve
cache-scheduling decisions in diffusion model inference.

Supports: flux, hunyuan_video, wan2.1
"""
import logging
import torch
from torch import Tensor
from typing import Optional, Dict, Any
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding
import comfy.ldm.common_dit
import comfy.model_management as mm

from .sea_utils import apply_sea_filter, rel_l1

logger = logging.getLogger("SeaCache")


# ===================== FLUX Forward =====================

def seacache_flux_forward(
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
    sc_opts = transformer_options.get("seacache_options", {})

    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # Initialize SeaCache state
    if not hasattr(self, '_sc_state') or self._sc_state.get('num_steps', 0) == 0:
        self._sc_state = {
            'cnt': 0,
            'num_steps': sc_opts.get('num_steps', 50),
            'seacache_thresh': sc_opts.get('seacache_thresh', 0.3),
            'power_exp': sc_opts.get('power_exp', 2.0),
            'accumulated_rel_l1': 0.0,
            'previous_modulated_input': None,
            'previous_residual': None,
            'sigmas': sc_opts.get('sigmas', None),
            'ret_steps': sc_opts.get('ret_steps', 1),
            'cutoff_steps': sc_opts.get('cutoff_steps', -1),
        }

    state = self._sc_state

    # Standard FLUX preamble
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    # SeaCache gating: use first block's norm1 modulation
    should_calc = True
    cnt = state['cnt']
    num_steps = state['num_steps']
    ret_steps = state['ret_steps']
    cutoff_steps = state['cutoff_steps'] if state['cutoff_steps'] >= 0 else num_steps - 1

    if cnt == 0 or cnt >= cutoff_steps or cnt < ret_steps or state['previous_modulated_input'] is None:
        should_calc = True
        state['accumulated_rel_l1'] = 0.0
    else:
        # Compute modulated input from first block for distance check
        modulated_inp = self.double_blocks[0].img_norm1(img)
        # Apply SEA filtering (2D spatial for images)
        # Reshape to spatial if possible using img_ids
        if img_ids is not None and img_ids.ndim >= 2:
            try:
                h = int(img_ids[:, :, 1].max().item() + 1) if img_ids.shape[-1] > 1 else 1
                w = int(img_ids[:, :, 2].max().item() + 1) if img_ids.shape[-1] > 2 else 1
                if h > 1 and w > 1:
                    modulated_inp_spatial = modulated_inp.reshape(
                        modulated_inp.shape[0], h, w, modulated_inp.shape[-1]
                    )
                    modulated_inp_spatial = apply_sea_filter(
                        modulated_inp_spatial,
                        state['sigmas'],
                        cnt,
                        power_exp=state['power_exp'],
                        dims=(-2, -3),
                        norm_mode="mean",
                    )
                    modulated_inp = modulated_inp_spatial.reshape(
                        modulated_inp.shape[0], -1, modulated_inp.shape[-1]
                    )
            except Exception:
                pass  # Fallback: use unfiltered modulated_inp

        state['accumulated_rel_l1'] += rel_l1(modulated_inp, state['previous_modulated_input'])
        if state['accumulated_rel_l1'] < state['seacache_thresh']:
            should_calc = False
        else:
            should_calc = True
            state['accumulated_rel_l1'] = 0.0

    # Update previous_modulated_input
    if should_calc or state['previous_modulated_input'] is None:
        modulated_inp = self.double_blocks[0].img_norm1(img)
        if img_ids is not None and img_ids.ndim >= 2:
            try:
                h = int(img_ids[:, :, 1].max().item() + 1) if img_ids.shape[-1] > 1 else 1
                w = int(img_ids[:, :, 2].max().item() + 1) if img_ids.shape[-1] > 2 else 1
                if h > 1 and w > 1:
                    modulated_inp_spatial = modulated_inp.reshape(
                        modulated_inp.shape[0], h, w, modulated_inp.shape[-1]
                    )
                    modulated_inp_spatial = apply_sea_filter(
                        modulated_inp_spatial,
                        state['sigmas'],
                        cnt,
                        power_exp=state['power_exp'],
                        dims=(-2, -3),
                        norm_mode="mean",
                    )
                    modulated_inp = modulated_inp_spatial.reshape(
                        modulated_inp.shape[0], -1, modulated_inp.shape[-1]
                    )
            except Exception:
                pass
        state['previous_modulated_input'] = modulated_inp.detach()

    blocks_replace = patches_replace.get("dit", {})

    # Main compute or cache reuse
    if not should_calc and state['previous_residual'] is not None:
        img = img + state['previous_residual']
    else:
        ori_img = img.clone()

        # Double stream blocks
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"],
                                                   vec=args["vec"], pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask},
                    {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        # Single stream blocks
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out
                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask},
                    {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add

        img = img[:, txt.shape[1]:, ...]

        # Cache residual
        state['previous_residual'] = (img - ori_img).detach()

    img = self.final_layer(img, vec)

    state['cnt'] += 1
    if state['cnt'] >= num_steps:
        state['cnt'] = 0

    return img


# ===================== HunyuanVideo Forward =====================

def seacache_hunyuanvideo_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    txt_mask: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    guiding_frame_index=None,
    ref_latent=None,
    control=None,
    transformer_options={},
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    sc_opts = transformer_options.get("seacache_options", {})

    # Initialize SeaCache state
    if not hasattr(self, '_sc_state') or self._sc_state.get('num_steps', 0) == 0:
        self._sc_state = {
            'cnt': 0,
            'num_steps': sc_opts.get('num_steps', 50),
            'seacache_thresh': sc_opts.get('seacache_thresh', 0.2),
            'power_exp': sc_opts.get('power_exp', 3.0),
            'accumulated_rel_l1': 0.0,
            'previous_modulated_input': None,
            'previous_residual': None,
            'sigmas': sc_opts.get('sigmas', None),
            'ret_steps': sc_opts.get('ret_steps', 1),
            'cutoff_steps': sc_opts.get('cutoff_steps', -1),
        }

    state = self._sc_state

    initial_shape = list(img.shape)
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

    if ref_latent is not None:
        ref_latent_ids = self.img_ids(ref_latent)
        ref_latent = self.img_in(ref_latent)
        img = torch.cat([ref_latent, img], dim=-2)
        ref_latent_ids[..., 0] = -1
        ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
        img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

    if guiding_frame_index is not None:
        token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
        vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
        vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
        frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
        modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
        modulation_dims_txt = [(0, None, 1)]
    else:
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        modulation_dims = None
        modulation_dims_txt = None

    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if txt_mask is not None and not torch.is_floating_point(txt_mask):
        txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

    txt = self.txt_in(txt, timesteps, txt_mask)

    ids = torch.cat((img_ids, txt_ids), dim=1)
    pe = self.pe_embedder(ids)

    img_len = img.shape[1]
    if txt_mask is not None:
        attn_mask_len = img_len + txt.shape[1]
        attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
        attn_mask[:, 0, img_len:] = txt_mask
    else:
        attn_mask = None

    # SeaCache gating
    should_calc = True
    cnt = state['cnt']
    num_steps = state['num_steps']
    ret_steps = state['ret_steps']
    cutoff_steps = state['cutoff_steps'] if state['cutoff_steps'] >= 0 else num_steps - 1

    if cnt == 0 or cnt >= cutoff_steps or cnt < ret_steps or state['previous_modulated_input'] is None:
        should_calc = True
        state['accumulated_rel_l1'] = 0.0
    else:
        # Use first double block's img_norm1 for distance computation
        modulated_inp = self.double_blocks[0].img_norm1(img)
        # Apply SEA filter (3D: T, H, W for video)
        # Derive spatial dims from patch_size and initial_shape
        try:
            tt = initial_shape[2] // self.patch_size[0]
            th = initial_shape[3] // self.patch_size[1]
            tw = initial_shape[4] // self.patch_size[2]
            if tt > 1 and th > 1 and tw > 1:
                modulated_inp_5d = modulated_inp.reshape(
                    modulated_inp.shape[0], tt, th, tw, modulated_inp.shape[-1]
                )
                modulated_inp_5d = apply_sea_filter(
                    modulated_inp_5d,
                    state['sigmas'],
                    cnt,
                    power_exp=state['power_exp'],
                    dims=(-2, -3, -4),
                    norm_mode="mean",
                )
                modulated_inp = modulated_inp_5d.reshape(
                    modulated_inp.shape[0], -1, modulated_inp.shape[-1]
                )
        except Exception:
            pass

        state['accumulated_rel_l1'] += rel_l1(modulated_inp, state['previous_modulated_input'])
        if state['accumulated_rel_l1'] < state['seacache_thresh']:
            should_calc = False
        else:
            should_calc = True
            state['accumulated_rel_l1'] = 0.0

    # Update previous modulated input
    if should_calc or state['previous_modulated_input'] is None:
        modulated_inp = self.double_blocks[0].img_norm1(img)
        try:
            tt = initial_shape[2] // self.patch_size[0]
            th = initial_shape[3] // self.patch_size[1]
            tw = initial_shape[4] // self.patch_size[2]
            if tt > 1 and th > 1 and tw > 1:
                modulated_inp_5d = modulated_inp.reshape(
                    modulated_inp.shape[0], tt, th, tw, modulated_inp.shape[-1]
                )
                modulated_inp_5d = apply_sea_filter(
                    modulated_inp_5d,
                    state['sigmas'],
                    cnt,
                    power_exp=state['power_exp'],
                    dims=(-2, -3, -4),
                    norm_mode="mean",
                )
                modulated_inp = modulated_inp_5d.reshape(
                    modulated_inp.shape[0], -1, modulated_inp.shape[-1]
                )
        except Exception:
            pass
        state['previous_modulated_input'] = modulated_inp.detach()

    blocks_replace = patches_replace.get("dit", {})

    if not should_calc and state['previous_residual'] is not None:
        img = img + state['previous_residual']
    else:
        ori_img = img.clone()

        # Double stream blocks
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"],
                                                   pe=args["pe"], attn_mask=args["attention_mask"],
                                                   modulation_dims_img=args["modulation_dims_img"],
                                                   modulation_dims_txt=args["modulation_dims_txt"])
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask,
                     'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt},
                    {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask,
                                 modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((img, txt), 1)

        # Single stream blocks
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"],
                                       attn_mask=args["attention_mask"],
                                       modulation_dims=args["modulation_dims"])
                    return out
                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask,
                     'modulation_dims': modulation_dims},
                    {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, :img_len] += add

        img = img[:, :img_len]

        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]

        # Cache residual (before final_layer)
        state['previous_residual'] = (img - ori_img).detach()

    img = self.final_layer(img, vec, modulation_dims=modulation_dims)

    shape = initial_shape[-3:]
    for i in range(len(shape)):
        shape[i] = shape[i] // self.patch_size[i]
    img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
    img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
    img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])

    state['cnt'] += 1
    if state['cnt'] >= num_steps:
        state['cnt'] = 0

    return img


# ===================== Wan2.1 Forward =====================

def seacache_wanmodel_forward(
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
    sc_opts = transformer_options.get("seacache_options", {})

    # Initialize SeaCache state
    if not hasattr(self, '_sc_state') or self._sc_state.get('num_steps', 0) == 0:
        self._sc_state = {
            'cnt': 0,
            'num_steps': sc_opts.get('num_steps', 50),
            'seacache_thresh': sc_opts.get('seacache_thresh', 0.2),
            'power_exp': sc_opts.get('power_exp', 3.0),
            'accumulated_rel_l1_even': 0.0,
            'accumulated_rel_l1_odd': 0.0,
            'previous_modulated_even': None,
            'previous_modulated_odd': None,
            'previous_residual_even': None,
            'previous_residual_odd': None,
            'sigmas': sc_opts.get('sigmas', None),
            'ret_steps': sc_opts.get('ret_steps', 1),
            'cutoff_steps': sc_opts.get('cutoff_steps', -1),
        }

    state = self._sc_state

    # Embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # Time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # Context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    # SeaCache gating (Wan uses CFG with cond/uncond split)
    cnt = state['cnt']
    num_steps = state['num_steps']
    ret_steps = state['ret_steps'] * 2  # *2 for cond+uncond
    cutoff_steps = (state['cutoff_steps'] * 2) if state['cutoff_steps'] >= 0 else num_steps - 2

    is_even = (cnt % 2 == 0)  # even=cond, odd=uncond

    # Compute modulated input for gating
    e_ = (self.blocks[0].modulation(e0)).chunk(6, dim=1)
    modulated_inp = self.blocks[0].norm1(x).float() * (1 + e_[1]) + e_[0]

    should_calc = True
    if cnt < ret_steps or cnt >= cutoff_steps:
        should_calc = True
        if is_even:
            state['accumulated_rel_l1_even'] = 0.0
        else:
            state['accumulated_rel_l1_odd'] = 0.0
    else:
        # Apply SEA filter in 3D (video: T, H, W)
        try:
            gt, gh, gw = grid_sizes[0], grid_sizes[1], grid_sizes[2]
            if gt > 1 and gh > 1 and gw > 1:
                modulated_inp_5d = modulated_inp.reshape(
                    modulated_inp.shape[0], gt, gh, gw, modulated_inp.shape[-1]
                )
                modulated_inp_5d = apply_sea_filter(
                    modulated_inp_5d,
                    state['sigmas'],
                    cnt // 2,
                    power_exp=state['power_exp'],
                    dims=(-2, -3, -4),
                    norm_mode="mean",
                )
                modulated_inp = modulated_inp_5d.reshape(
                    modulated_inp.shape[0], -1, modulated_inp.shape[-1]
                )
        except Exception:
            pass

        if is_even:
            if state['previous_modulated_even'] is not None:
                state['accumulated_rel_l1_even'] += rel_l1(modulated_inp, state['previous_modulated_even'])
                if state['accumulated_rel_l1_even'] < state['seacache_thresh']:
                    should_calc = False
                else:
                    should_calc = True
                    state['accumulated_rel_l1_even'] = 0.0
        else:
            if state['previous_modulated_odd'] is not None:
                state['accumulated_rel_l1_odd'] += rel_l1(modulated_inp, state['previous_modulated_odd'])
                if state['accumulated_rel_l1_odd'] < state['seacache_thresh']:
                    should_calc = False
                else:
                    should_calc = True
                    state['accumulated_rel_l1_odd'] = 0.0

    # Store current modulated input
    if is_even:
        state['previous_modulated_even'] = modulated_inp.detach()
    else:
        state['previous_modulated_odd'] = modulated_inp.detach()

    blocks_replace = patches_replace.get("dit", {})

    # Main compute or reuse
    if not should_calc:
        residual = state['previous_residual_even'] if is_even else state['previous_residual_odd']
        if residual is not None:
            x = x + residual
        else:
            should_calc = True

    if should_calc:
        ori_x = x.clone()

        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"],
                                       freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": x, "txt": context, "vec": e0, "pe": freqs},
                    {"original_block": block_wrap, "transformer_options": transformer_options})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        # Cache residual
        if is_even:
            state['previous_residual_even'] = (x - ori_x).detach()
        else:
            state['previous_residual_odd'] = (x - ori_x).detach()

    # Head
    x = self.head(x, e)

    # Unpatchify
    x = self.unpatchify(x, grid_sizes)

    state['cnt'] += 1
    if state['cnt'] >= num_steps:
        state['cnt'] = 0

    return x


# ===================== Apply SeaCache =====================

SEACACHE_MODEL_TYPES = ["flux", "hunyuan_video", "wan2.1"]

# Default parameters per model type
SEACACHE_MODEL_DEFAULTS = {
    "flux": {"seacache_thresh": 0.3, "power_exp": 2.0, "ret_steps": 1},
    "hunyuan_video": {"seacache_thresh": 0.2, "power_exp": 3.0, "ret_steps": 1},
    "wan2.1": {"seacache_thresh": 0.2, "power_exp": 3.0, "ret_steps": 1},
}


class SeaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (SEACACHE_MODEL_TYPES, {"default": "flux"}),
                "seacache_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                                              "tooltip": "Cache threshold. Higher = more skip = faster but lower quality. FLUX: 0.3~2x, 0.6~3x; Wan: 0.1~2x, 0.2~3x"}),
                "power_exp": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.5,
                                        "tooltip": "Power spectrum exponent. 2.0 for images, 3.0 for video."}),
                "ret_steps": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1,
                                      "tooltip": "Number of initial retention steps (always full compute)."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_seacache"
    CATEGORY = "SeaCache"
    TITLE = "SeaCache"

    def apply_seacache(self, model, model_type: str, seacache_thresh: float,
                       power_exp: float, ret_steps: int):
        new_model = model.clone()

        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}

        new_model.model_options["transformer_options"]["seacache_options"] = {
            "seacache_thresh": seacache_thresh,
            "power_exp": power_exp,
            "ret_steps": ret_steps,
        }

        diffusion_model = new_model.get_model_object("diffusion_model")

        # Clean up any existing state
        if hasattr(diffusion_model, '_sc_state'):
            delattr(diffusion_model, '_sc_state')

        # Select forward function based on model type
        if model_type == "flux":
            forward_fn = seacache_flux_forward
            forward_attr = "forward_orig"
        elif model_type == "hunyuan_video":
            forward_fn = seacache_hunyuanvideo_forward
            forward_attr = "forward_orig"
        elif model_type == "wan2.1":
            forward_fn = seacache_wanmodel_forward
            forward_attr = "forward_orig"
        else:
            raise ValueError(f"Unsupported SeaCache model type: {model_type}")

        context = patch.multiple(
            diffusion_model,
            **{forward_attr: forward_fn.__get__(diffusion_model, diffusion_model.__class__)}
        )

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]

            # Pass num_steps and sigmas from sample_sigmas
            sigmas = c["transformer_options"].get("sample_sigmas")
            if sigmas is not None:
                num_steps = len(sigmas) - 1
                c["transformer_options"]["seacache_options"]["num_steps"] = num_steps
                c["transformer_options"]["seacache_options"]["sigmas"] = sigmas

                # Reset state on first step
                matched_step_index = (sigmas == timestep[0]).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    current_step_index = 0
                    for i in range(len(sigmas) - 1):
                        if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                            current_step_index = i
                            break

                if current_step_index == 0:
                    if hasattr(diffusion_model, '_sc_state'):
                        delattr(diffusion_model, '_sc_state')

            with context:
                result = model_function(input, timestep, **c)

            return result

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)


def apply_seacache(model, model_type: str, seacache_thresh: float,
                   power_exp: float, ret_steps: int):
    """Convenience function for use from the unified node."""
    sc = SeaCache()
    return sc.apply_seacache(model, model_type, seacache_thresh, power_exp, ret_steps)[0]
