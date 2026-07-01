"""
TaylorSeer / HiCache model acceleration for ComfyUI.
Based on:
- TaylorSeer: https://github.com/Shenyi-Z/TaylorSeer (ICCV 2025)
- HiCache: https://arxiv.org/abs/2508.16984 (ICLR 2026)

Supports: flux, hunyuan_video, wan2.1, hidream
Prediction modes:
- taylor: Standard Taylor series expansion (TaylorSeer)
- hicache: Hermite polynomial-based prediction (HiCache)
- taylor_scaled: Taylor with dual-scaling trick
"""
import logging
import torch
from torch import Tensor
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
import comfy.ldm.common_dit
import comfy.model_management as mm

from .taylor_utils import (
    derivative_approximation,
    taylor_formula,
    taylor_cache_init,
    cal_type,
    cache_init,
)

logger = logging.getLogger("TaylorSeer")


# ===================== FLUX Forward =====================

def taylorseer_flux_forward(
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
    ts_opts = transformer_options.get("taylorseer_options", {})

    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # Initialize cache on first call of a new generation
    if not hasattr(self, '_ts_cache_dic') or self._ts_current['step'] >= self._ts_current['num_steps']:
        num_double = len(self.double_blocks)
        num_single = len(self.single_blocks)
        self._ts_cache_dic, self._ts_current = cache_init(
            num_double_layers=num_double,
            num_single_layers=num_single,
            num_steps=ts_opts.get('num_steps', 50),
            max_order=ts_opts.get('max_order', 1),
            fresh_threshold=ts_opts.get('fresh_threshold', 6),
            first_enhance=ts_opts.get('first_enhance', 3),
            prediction_mode=ts_opts.get('prediction_mode', 'taylor'),
            hicache_scale_factor=ts_opts.get('hicache_scale_factor', 0.5),
        )

    cache_dic = self._ts_cache_dic
    current = self._ts_current

    # Determine calculation type for this step
    cal_type(cache_dic, current)

    # Standard FLUX forward preamble
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

    blocks_replace = patches_replace.get("dit", {})

    # Double stream blocks
    current['stream'] = 'double_stream'
    for i, block in enumerate(self.double_blocks):
        current['layer'] = i

        if current['type'] == 'full':
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

            # Cache the block outputs for Taylor prediction
            current['module'] = 'img'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=img.clone())

            current['module'] = 'txt'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=txt.clone())

        elif current['type'] == 'Taylor':
            # Predict using Taylor expansion
            current['module'] = 'img'
            img = taylor_formula(cache_dic=cache_dic, current=current)
            current['module'] = 'txt'
            txt = taylor_formula(cache_dic=cache_dic, current=current)

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
    current['stream'] = 'single_stream'
    for i, block in enumerate(self.single_blocks):
        current['layer'] = i

        if current['type'] == 'full':
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

            current['module'] = 'total'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=img.clone())

        elif current['type'] == 'Taylor':
            current['module'] = 'total'
            img = taylor_formula(cache_dic=cache_dic, current=current)

        if control is not None:
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    img = img[:, txt.shape[1]:, ...]

    img = self.final_layer(img, vec)

    current['step'] += 1

    return img


# ===================== HunyuanVideo Forward =====================

def taylorseer_hunyuanvideo_forward(
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
    ts_opts = transformer_options.get("taylorseer_options", {})

    if not hasattr(self, '_ts_cache_dic') or self._ts_current['step'] >= self._ts_current['num_steps']:
        num_double = len(self.double_blocks)
        num_single = len(self.single_blocks)
        self._ts_cache_dic, self._ts_current = cache_init(
            num_double_layers=num_double,
            num_single_layers=num_single,
            num_steps=ts_opts.get('num_steps', 50),
            max_order=ts_opts.get('max_order', 1),
            fresh_threshold=ts_opts.get('fresh_threshold', 5),
            first_enhance=ts_opts.get('first_enhance', 1),
            prediction_mode=ts_opts.get('prediction_mode', 'taylor'),
            hicache_scale_factor=ts_opts.get('hicache_scale_factor', 0.5),
        )

    cache_dic = self._ts_cache_dic
    current = self._ts_current
    cal_type(cache_dic, current)

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

    blocks_replace = patches_replace.get("dit", {})

    # Double stream blocks
    current['stream'] = 'double_stream'
    for i, block in enumerate(self.double_blocks):
        current['layer'] = i

        if current['type'] == 'full':
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

            current['module'] = 'img'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=img.clone())
            current['module'] = 'txt'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=txt.clone())

        elif current['type'] == 'Taylor':
            current['module'] = 'img'
            img = taylor_formula(cache_dic=cache_dic, current=current)
            current['module'] = 'txt'
            txt = taylor_formula(cache_dic=cache_dic, current=current)

        if control is not None:
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((img, txt), 1)

    # Single stream blocks
    current['stream'] = 'single_stream'
    for i, block in enumerate(self.single_blocks):
        current['layer'] = i

        if current['type'] == 'full':
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

            current['module'] = 'total'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=img.clone())

        elif current['type'] == 'Taylor':
            current['module'] = 'total'
            img = taylor_formula(cache_dic=cache_dic, current=current)

        if control is not None:
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, :img_len] += add

    img = img[:, :img_len]

    if ref_latent is not None:
        img = img[:, ref_latent.shape[1]:]

    img = self.final_layer(img, vec, modulation_dims=modulation_dims)

    shape = initial_shape[-3:]
    for i in range(len(shape)):
        shape[i] = shape[i] // self.patch_size[i]
    img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
    img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
    img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])

    current['step'] += 1

    return img


# ===================== Wan2.1 Forward =====================

def taylorseer_wanmodel_forward(
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
    ts_opts = transformer_options.get("taylorseer_options", {})

    if not hasattr(self, '_ts_cache_dic') or self._ts_current['step'] >= self._ts_current['num_steps']:
        num_blocks = len(self.blocks)
        self._ts_cache_dic, self._ts_current = cache_init(
            num_double_layers=num_blocks,
            num_single_layers=0,
            num_steps=ts_opts.get('num_steps', 50),
            max_order=ts_opts.get('max_order', 1),
            fresh_threshold=ts_opts.get('fresh_threshold', 5),
            first_enhance=ts_opts.get('first_enhance', 1),
            prediction_mode=ts_opts.get('prediction_mode', 'taylor'),
            hicache_scale_factor=ts_opts.get('hicache_scale_factor', 0.5),
        )

    cache_dic = self._ts_cache_dic
    current = self._ts_current
    cal_type(cache_dic, current)

    # Embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # Time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # Context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    blocks_replace = patches_replace.get("dit", {})

    # Blocks
    current['stream'] = 'double_stream'
    for i, block in enumerate(self.blocks):
        current['layer'] = i

        if current['type'] == 'full':
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

            current['module'] = 'total'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=x.clone())

        elif current['type'] == 'Taylor':
            current['module'] = 'total'
            x = taylor_formula(cache_dic=cache_dic, current=current)

    # Head
    x = self.head(x, e)

    # Unpatchify
    x = self.unpatchify(x, grid_sizes)

    current['step'] += 1

    return x


# ===================== HiDream Forward =====================

def taylorseer_hidream_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    y=None,
    context=None,
    encoder_hidden_states_llama3=None,
    image_cond=None,
    control=None,
    transformer_options={},
) -> torch.Tensor:
    from einops import repeat

    ts_opts = transformer_options.get("taylorseer_options", {})

    if not hasattr(self, '_ts_cache_dic') or self._ts_current['step'] >= self._ts_current['num_steps']:
        num_double = len(self.double_stream_blocks)
        num_single = len(self.single_stream_blocks)
        self._ts_cache_dic, self._ts_current = cache_init(
            num_double_layers=num_double,
            num_single_layers=num_single,
            num_steps=ts_opts.get('num_steps', 50),
            max_order=ts_opts.get('max_order', 2),
            fresh_threshold=ts_opts.get('fresh_threshold', 4),
            first_enhance=ts_opts.get('first_enhance', 1),
            prediction_mode=ts_opts.get('prediction_mode', 'taylor'),
            hicache_scale_factor=ts_opts.get('hicache_scale_factor', 0.5),
        )

    cache_dic = self._ts_cache_dic
    current = self._ts_current
    cal_type(cache_dic, current)

    bs, c, h, w = x.shape
    if image_cond is not None:
        x = torch.cat([x, image_cond], dim=-1)
    hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    timesteps = t
    pooled_embeds = y
    T5_encoder_hidden_states = context

    img_sizes = None
    batch_size = hidden_states.shape[0]
    hidden_states_type = hidden_states.dtype

    timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
    timesteps = self.t_embedder(timesteps, hidden_states_type)
    p_embedder = self.p_embedder(pooled_embeds)
    adaln_input = timesteps + p_embedder

    hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
    if image_tokens_masks is None:
        pH, pW = img_sizes[0]
        img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    hidden_states = self.x_embedder(hidden_states)

    encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
    encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

    if self.caption_projection is not None:
        new_encoder_hidden_states = []
        for i, enc_hidden_state in enumerate(encoder_hidden_states):
            enc_hidden_state = self.caption_projection[i](enc_hidden_state)
            enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
            new_encoder_hidden_states.append(enc_hidden_state)
        encoder_hidden_states = new_encoder_hidden_states
        T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
        T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states.append(T5_encoder_hidden_states)

    txt_ids = torch.zeros(
        batch_size,
        encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
        3,
        device=img_ids.device, dtype=img_ids.dtype
    )
    ids = torch.cat((img_ids, txt_ids), dim=1)
    rope = self.pe_embedder(ids)

    # Double stream blocks
    current['stream'] = 'double_stream'
    block_id = 0
    initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
    initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]

    for bid, block in enumerate(self.double_stream_blocks):
        current['layer'] = bid

        if current['type'] == 'full':
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states, initial_encoder_hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=cur_encoder_hidden_states,
                adaln_input=adaln_input,
                rope=rope,
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]

            current['module'] = 'hidden'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states.clone())
            current['module'] = 'enc'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=initial_encoder_hidden_states.clone())

        elif current['type'] == 'Taylor':
            current['module'] = 'hidden'
            hidden_states = taylor_formula(cache_dic=cache_dic, current=current)
            current['module'] = 'enc'
            initial_encoder_hidden_states = taylor_formula(cache_dic=cache_dic, current=current)

        block_id += 1

    # Single stream blocks
    current['stream'] = 'single_stream'
    image_tokens_seq_len = hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
    hidden_states_seq_len = hidden_states.shape[1]
    if image_tokens_masks is not None:
        encoder_attention_mask_ones = torch.ones(
            (batch_size, initial_encoder_hidden_states.shape[1] + encoder_hidden_states[block_id].shape[1]),
            device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
        )
        image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

    for bid, block in enumerate(self.single_stream_blocks):
        current['layer'] = bid

        if current['type'] == 'full':
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=None,
                adaln_input=adaln_input,
                rope=rope,
            )
            hidden_states = hidden_states[:, :hidden_states_seq_len]

            current['module'] = 'total'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states.clone())

        elif current['type'] == 'Taylor':
            current['module'] = 'total'
            hidden_states = taylor_formula(cache_dic=cache_dic, current=current)

        block_id += 1

    hidden_states = hidden_states[:, :image_tokens_seq_len, ...]

    output = self.final_layer(hidden_states, adaln_input)
    output = self.unpatchify(output, img_sizes)

    current['step'] += 1

    return -output[:, :, :h, :w]


# ===================== Apply TaylorSeer =====================

# Model type -> default parameters
TAYLORSEER_MODEL_DEFAULTS = {
    "flux": {"max_order": 1, "fresh_threshold": 6, "first_enhance": 3},
    "hunyuan_video": {"max_order": 1, "fresh_threshold": 5, "first_enhance": 1},
    "wan2.1": {"max_order": 1, "fresh_threshold": 5, "first_enhance": 1},
    "hidream": {"max_order": 2, "fresh_threshold": 4, "first_enhance": 1},
}

# HiCache recommended defaults (uses lower max_order with Hermite basis)
HICACHE_MODEL_DEFAULTS = {
    "flux": {"max_order": 1, "fresh_threshold": 6, "first_enhance": 3, "hicache_scale_factor": 0.5},
    "hunyuan_video": {"max_order": 1, "fresh_threshold": 5, "first_enhance": 1, "hicache_scale_factor": 0.5},
    "wan2.1": {"max_order": 1, "fresh_threshold": 5, "first_enhance": 1, "hicache_scale_factor": 0.5},
    "hidream": {"max_order": 1, "fresh_threshold": 4, "first_enhance": 1, "hicache_scale_factor": 0.5},
}

TAYLORSEER_MODEL_TYPES = list(TAYLORSEER_MODEL_DEFAULTS.keys())

PREDICTION_MODES = ["taylor", "hicache", "taylor_scaled"]


class TaylorSeer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (TAYLORSEER_MODEL_TYPES, {"default": "flux"}),
                "max_order": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1,
                                      "tooltip": "Taylor expansion max order. Higher = more accurate but more memory."}),
                "fresh_threshold": ("INT", {"default": 6, "min": 2, "max": 20, "step": 1,
                                            "tooltip": "Interval between full computation steps. Higher = faster but less quality."}),
                "first_enhance": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1,
                                          "tooltip": "Number of initial steps that always do full computation."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_taylorseer"
    CATEGORY = "TaylorSeer"
    TITLE = "TaylorSeer"

    def apply_taylorseer(self, model, model_type: str, max_order: int,
                         fresh_threshold: int, first_enhance: int,
                         prediction_mode: str = "taylor",
                         hicache_scale_factor: float = 0.5):
        new_model = model.clone()

        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}

        new_model.model_options["transformer_options"]["taylorseer_options"] = {
            "max_order": max_order,
            "fresh_threshold": fresh_threshold,
            "first_enhance": first_enhance,
            "prediction_mode": prediction_mode,
            "hicache_scale_factor": hicache_scale_factor,
        }

        diffusion_model = new_model.get_model_object("diffusion_model")

        # Clean up any existing state
        if hasattr(diffusion_model, '_ts_cache_dic'):
            delattr(diffusion_model, '_ts_cache_dic')
        if hasattr(diffusion_model, '_ts_current'):
            delattr(diffusion_model, '_ts_current')

        # Select forward function based on model type
        if model_type == "flux":
            forward_fn = taylorseer_flux_forward
            forward_attr = "forward_orig"
        elif model_type == "hunyuan_video":
            forward_fn = taylorseer_hunyuanvideo_forward
            forward_attr = "forward_orig"
        elif model_type == "wan2.1":
            forward_fn = taylorseer_wanmodel_forward
            forward_attr = "forward_orig"
        elif model_type == "hidream":
            forward_fn = taylorseer_hidream_forward
            forward_attr = "forward"
        else:
            raise ValueError(f"Unsupported TaylorSeer model type: {model_type}")

        context = patch.multiple(
            diffusion_model,
            **{forward_attr: forward_fn.__get__(diffusion_model, diffusion_model.__class__)}
        )

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]

            # Pass num_steps from sigmas
            sigmas = c["transformer_options"].get("sample_sigmas")
            if sigmas is not None:
                num_steps = len(sigmas) - 1
                c["transformer_options"]["taylorseer_options"]["num_steps"] = num_steps

                # Reset cache on first step
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
                    if hasattr(diffusion_model, '_ts_cache_dic'):
                        delattr(diffusion_model, '_ts_cache_dic')
                    if hasattr(diffusion_model, '_ts_current'):
                        delattr(diffusion_model, '_ts_current')

            with context:
                result = model_function(input, timestep, **c)

            return result

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)


def apply_taylorseer(model, model_type: str, max_order: int, fresh_threshold: int, first_enhance: int,
                     prediction_mode: str = "taylor", hicache_scale_factor: float = 0.5):
    """Convenience function for use from the unified node."""
    ts = TaylorSeer()
    return ts.apply_taylorseer(model, model_type, max_order, fresh_threshold, first_enhance,
                               prediction_mode=prediction_mode,
                               hicache_scale_factor=hicache_scale_factor)[0]
