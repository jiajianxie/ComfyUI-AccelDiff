"""
ComfyUI节点 - 动态参数输入
支持独立选择 sampler_method 和 model_method，可单独或同时使用
"""

import comfy

support_sampler = ["euler", "heun", "heunpp2","dpm_2","lms", "dpm_fast", "dpm_adaptive",
                 "dpmpp_2m", "ipndm", "ipndm_v", "deis", "res_multistep",
                 "gradient_estimation","ddim", "euler_ancestral","dpm_2_ancestral"]

# SADA supports a subset of samplers
SADA_SUPPORT_SAMPLER = ["euler", "heun", "heunpp2", "dpm_2", "lms", "dpmpp_2m", "ddim",
                        "euler_ancestral", "dpm_2_ancestral"]

TEACACHE_MODEL_TYPES = [
    "flux", "flux-kontext", "ltxv", "lumina_2", "hunyuan_video",
    "hidream_i1_full", "hidream_i1_dev", "hidream_i1_fast",
    "wan2.1_t2v_1.3B", "wan2.1_t2v_14B",
    "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B",
    "wan2.1_t2v_1.3B_ret_mode", "wan2.1_t2v_14B_ret_mode",
    "wan2.1_i2v_480p_14B_ret_mode", "wan2.1_i2v_720p_14B_ret_mode",
]

MAGCACHE_MODEL_TYPES = [
    "flux", "flux_kontext", "chroma", "qwen_image",
    "hunyuan_video", "hunyuan_video1.5",
    "wan2.1_t2v_1.3B", "wan2.1_t2v_14B",
    "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B",
    "wan2.1_vace_1.3B", "wan2.1_vace_14B",
]

TAYLORSEER_MODEL_TYPES = ["flux", "hunyuan_video", "wan2.1", "hidream"]

HICACHE_MODEL_TYPES = ["flux", "hunyuan_video", "wan2.1", "hidream"]
HICACHE_PREDICTION_MODES = ["hicache", "taylor_scaled"]

SEACACHE_MODEL_TYPES = ["flux", "hunyuan_video", "wan2.1"]

# ========== sampler_method 可选项 ==========
SAMPLER_METHOD_LIST = ["None", "AdaptiveDiff", "EasyCache", "SADA", "ZEUS"]

ZEUS_INTERP_MODES = ["psi", "x_0"]
ZEUS_CACHING_MODES = ["reuse_interp", "interp_all", "reuse_all"]

# ========== model_method (model-feature) 可选项 ==========
MODEL_METHOD_LIST = ["None", "TeaCache", "MagCache", "TaylorSeer", "HiCache", "SeaCache"]


def _apply_teacache(model, model_type, rel_l1_thresh, start_percent, end_percent, cache_device):
    from .model.teacache.nodes import TeaCache
    return TeaCache().apply_teacache(
        model, model_type, rel_l1_thresh, start_percent, end_percent, cache_device
    )[0]


def _apply_taylorseer(model, model_type, max_order, fresh_threshold, first_enhance):
    from .model.taylorseer.nodes import apply_taylorseer
    return apply_taylorseer(model, model_type, max_order, fresh_threshold, first_enhance)


def _apply_hicache(model, model_type, max_order, fresh_threshold, first_enhance,
                   prediction_mode, hicache_scale_factor):
    from .model.taylorseer.nodes import apply_taylorseer
    return apply_taylorseer(model, model_type, max_order, fresh_threshold, first_enhance,
                            prediction_mode=prediction_mode,
                            hicache_scale_factor=hicache_scale_factor)


def _apply_seacache(model, model_type, seacache_thresh, power_exp, ret_steps):
    from .model.seacache.nodes import apply_seacache
    return apply_seacache(model, model_type, seacache_thresh, power_exp, ret_steps)


def _apply_magcache(model, model_type, magcache_thresh, retention_ratio, magcache_K, start_step, end_step):
    from .model.magcache.nodes import MagCache
    return MagCache().apply_magcache(
        model, model_type, magcache_thresh, retention_ratio, magcache_K, start_step, end_step
    )[0]



class DynamicModelParamsNode:
    """
    支持独立选择 sampler_method 和 model_method 的统一加速节点。
    - sampler_method: 选择 sampler 加速方法（SADA, AdaptiveDiff, EasyCache, ZEUS），None 表示不使用
    - model_method: 选择 model-feature 加速方法（TeaCache, MagCache 等），None 表示不使用
    - 两者可独立使用，也可同时使用
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_method": (SAMPLER_METHOD_LIST, {"default": "None"}),
                "model_method": (MODEL_METHOD_LIST, {"default": "None"}),
            },
            "optional": {
                # 具体的采样器选择 (euler, heun, etc.)
                "sampler_name": (support_sampler,),
                # MODEL 输入
                "model": ("MODEL",),
                # === AdaptiveDiff sampler 参数 ===
                "max_skip_steps": ("INT", {"default": 4, "min": 0, "max": 10, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                # === EasyCache sampler 参数 ===
                "ec_threshold": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001}),
                "ret_steps": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                # === SADA sampler 参数 ===
                "lagrange_term": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1}),
                "lagrange_int": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "lagrange_step": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "max_interval": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "acc_start": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "acc_end": ("INT", {"default": 47, "min": -100, "max": 200, "step": 1}),

                # === TeaCache model 参数 ===
                "teacache_model_type": (TEACACHE_MODEL_TYPES, {"default": "wan2.1_t2v_1.3B_ret_mode"}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache_device": (["cuda", "cpu"], {"default": "cuda"}),
                # === MagCache model 参数 ===
                "magcache_model_type": (MAGCACHE_MODEL_TYPES, {"default": "wan2.1_t2v_1.3B"}),
                "magcache_thresh": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01}),
                "retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.3, "step": 0.01}),
                "magcache_K": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "end_step": ("INT", {"default": -1, "min": -100, "max": 100, "step": 1}),
                # === TaylorSeer model 参数 ===
                "taylorseer_model_type": (TAYLORSEER_MODEL_TYPES, {"default": "flux"}),
                "max_order": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
                "fresh_threshold": ("INT", {"default": 6, "min": 2, "max": 20, "step": 1}),
                "first_enhance": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                # === HiCache model 参数 ===
                "hicache_model_type": (HICACHE_MODEL_TYPES, {"default": "flux"}),
                "hicache_prediction_mode": (HICACHE_PREDICTION_MODES, {"default": "hicache"}),
                "hicache_max_order": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
                "hicache_fresh_threshold": ("INT", {"default": 6, "min": 2, "max": 20, "step": 1}),
                "hicache_first_enhance": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "hicache_scale_factor": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01}),
                # === SeaCache model 参数 ===
                "seacache_model_type": (SEACACHE_MODEL_TYPES, {"default": "flux"}),
                "seacache_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seacache_power_exp": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.5}),
                "seacache_ret_steps": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
                # === ZEUS sampler 参数 (默认值来自 flux_demo.py) ===
                "zeus_denominator": ("INT", {"default": 3, "min": 2, "max": 6, "step": 1}),
                "zeus_modular": ("STRING", {"default": "0,1"}),
                "zeus_acc_start": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "zeus_acc_end": ("INT", {"default": 45, "min": -100, "max": 200, "step": 1}),
                "zeus_interp_mode": (ZEUS_INTERP_MODES, {"default": "psi"}),
                "zeus_caching_mode": (ZEUS_CACHING_MODES, {"default": "reuse_interp"}),
                "zeus_max_interval": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "zeus_lagrange_term": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1}),
                "zeus_lagrange_int": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "zeus_lagrange_step": ("INT", {"default": 24, "min": 5, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("SAMPLER", "MODEL")
    RETURN_NAMES = ("sampler", "model")
    FUNCTION = "get_sampler"
    CATEGORY = "AccelDiff"

    @classmethod
    def IS_CHANGED(cls, sampler_method, model_method, **kwargs):
        return f"{sampler_method}_{model_method}"

    def get_sampler(self, sampler_method="AdaptiveDiff", model_method="None",
                    sampler_name="euler",
                    model=None,
                    # --- AdaptiveDiff sampler 参数 ---
                    max_skip_steps=4,
                    threshold=0.01,
                    # --- EasyCache sampler 参数 ---
                    ec_threshold=0.025,
                    ret_steps=5,
                    # --- SADA sampler 参数 ---
                    lagrange_term=3,
                    lagrange_int=4,
                    lagrange_step=20,
                    max_interval=4,
                    acc_start=10,
                    acc_end=47,
                    # --- ZEUS sampler 参数 ---
                    zeus_denominator=3,
                    zeus_modular="0,1",
                    zeus_acc_start=10,
                    zeus_acc_end=45,
                    zeus_interp_mode="psi",
                    zeus_caching_mode="reuse_interp",
                    zeus_max_interval=4,
                    zeus_lagrange_term=3,
                    zeus_lagrange_int=6,
                    zeus_lagrange_step=24,
                    # --- TeaCache model 参数 ---
                    teacache_model_type="wan2.1_t2v_1.3B_ret_mode",
                    rel_l1_thresh=0.4,
                    start_percent=0.0,
                    end_percent=1.0,
                    cache_device="cuda",
                    # --- MagCache model 参数 ---
                    magcache_model_type="wan2.1_t2v_1.3B",
                    magcache_thresh=0.06,
                    retention_ratio=0.2,
                    magcache_K=2,
                    start_step=0,
                    end_step=-1,
                    # --- TaylorSeer model 参数 ---
                    taylorseer_model_type="flux",
                    max_order=1,
                    fresh_threshold=6,
                    first_enhance=3,
                    # --- HiCache model 参数 ---
                    hicache_model_type="flux",
                    hicache_prediction_mode="hicache",
                    hicache_max_order=1,
                    hicache_fresh_threshold=6,
                    hicache_first_enhance=3,
                    hicache_scale_factor=0.7,
                    # --- SeaCache model 参数 ---
                    seacache_model_type="flux",
                    seacache_thresh=0.3,
                    seacache_power_exp=2.0,
                    seacache_ret_steps=1,
                    **kwargs):

        out_sampler = None
        out_model = None

        # ====== Sampler 构建 ======
        if sampler_method != "None":
            extra_options = {}

            # --- 根据 sampler_method 导入对应的模块和 KSAMPLER ---
            if sampler_method == "SADA":
                from .sampler.sada import KSAMPLER as _KSAMPLER
                from .sampler import sada as _sampling
                _ksampler_kwargs = dict(
                    max_interval=max_interval,
                    acc_start=max(acc_start, 3),
                    acc_end=acc_end,
                    lagrange_term=lagrange_term,
                    lagrange_int=lagrange_int,
                    lagrange_step=lagrange_step,
                )
            elif sampler_method == "ZEUS":
                from .sampler.zeus import KSAMPLER as _KSAMPLER
                from .sampler import zeus as _sampling
                zeus_modular_tuple = tuple(int(x.strip()) for x in zeus_modular.split(",") if x.strip())
                _ksampler_kwargs = dict(
                    denominator=zeus_denominator,
                    modular=zeus_modular_tuple,
                    acc_start=zeus_acc_start,
                    acc_end=zeus_acc_end,
                    interp_mode=zeus_interp_mode,
                    caching_mode=zeus_caching_mode,
                    max_interval=zeus_max_interval,
                    lagrange_term=zeus_lagrange_term,
                    lagrange_int=zeus_lagrange_int,
                    lagrange_step=zeus_lagrange_step,
                )
            elif sampler_method == "AdaptiveDiff":
                from .sampler.adaptivediff import KSAMPLER as _KSAMPLER
                from .sampler import adaptivediff as _sampling
                _ksampler_kwargs = dict(
                    max_skip_steps=max_skip_steps,
                    threshold=threshold,
                )
            else:  # EasyCache
                from .sampler.easycache import KSAMPLER as _KSAMPLER
                from .sampler import easycache as _sampling
                _ksampler_kwargs = dict(
                    threshold=ec_threshold,
                    ret_steps=ret_steps,
                )

            # --- sampler_name 映射 ---
            if sampler_method == "SADA":
                _sampler_name = sampler_name if sampler_name in SADA_SUPPORT_SAMPLER else "euler"
            else:
                _sampler_name = "euler" if sampler_name == "ddim" else sampler_name
            _inpaint_opts = {"random": True} if sampler_name == "ddim" else {}

            # --- dpm_fast / dpm_adaptive 需要特殊包装（签名不同） ---
            if _sampler_name in ("dpm_fast", "dpm_adaptive") and sampler_method != "SADA":
                _raw_func = getattr(_sampling, "sample_{}".format(_sampler_name))
                _pass_keys = list(_ksampler_kwargs.keys())

                def _make_wrapper(raw_func, pass_keys):
                    def wrapper(model, noise, sigmas, extra_args=None, callback=None, disable=None, **kw):
                        if len(sigmas) <= 1:
                            return noise
                        sigma_min = sigmas[-1]
                        if sigma_min == 0:
                            sigma_min = sigmas[-2]
                        total_steps = len(sigmas) - 1
                        pass_kw = {k: kw.pop(k) for k in pass_keys if k in kw}
                        return raw_func(model, noise, sigma_min, sigmas[0], total_steps,
                                        extra_args=extra_args, callback=callback, disable=disable,
                                        **pass_kw)
                    return wrapper

                sampler_function = _make_wrapper(_raw_func, _pass_keys)
            else:
                sampler_function = getattr(_sampling, "sample_{}".format(_sampler_name))

            # --- 统一构造 KSAMPLER ---
            out_sampler = _KSAMPLER(
                sampler_function,
                extra_options=extra_options,
                inpaint_options=_inpaint_opts,
                **_ksampler_kwargs,
            )

        # ====== Model 构建 ======
        if model_method != "None":
            if model is None:
                raise ValueError(f"model_method={model_method} requires a MODEL input")

            if model_method == "TeaCache":
                out_model = _apply_teacache(
                    model, teacache_model_type, rel_l1_thresh,
                    start_percent, end_percent, cache_device,
                )
            elif model_method == "MagCache":
                out_model = _apply_magcache(
                    model, magcache_model_type, magcache_thresh,
                    retention_ratio, magcache_K, start_step, end_step,
                )
            elif model_method == "TaylorSeer":
                out_model = _apply_taylorseer(
                    model, taylorseer_model_type, max_order,
                    fresh_threshold, first_enhance,
                )
            elif model_method == "HiCache":
                out_model = _apply_hicache(
                    model, hicache_model_type, hicache_max_order,
                    hicache_fresh_threshold, hicache_first_enhance,
                    hicache_prediction_mode, hicache_scale_factor,
                )
            elif model_method == "SeaCache":
                out_model = _apply_seacache(
                    model, seacache_model_type, seacache_thresh,
                    seacache_power_exp, seacache_ret_steps,
                )
            else:
                out_model = model

        return (out_sampler, out_model)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AccelDiffUnified": DynamicModelParamsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AccelDiffUnified": "AccelDiff Unified"
}
