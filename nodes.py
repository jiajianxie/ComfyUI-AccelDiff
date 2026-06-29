"""
ComfyUI节点 - 动态参数输入
根据选择的模型名字动态展示不同数量的参数，返回 SAMPLER
"""

import comfy

support_sampler = ["euler", "heun", "heunpp2","dpm_2","lms", "dpm_fast", "dpm_adaptive",
                 "dpmpp_2m", "ipndm", "ipndm_v", "deis", "res_multistep",
                 "gradient_estimation","ddim",
                 "euler_ancestral","dpm_2_ancestral"]

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



def _apply_teacache(model, model_type, rel_l1_thresh, start_percent, end_percent, cache_device):
    from .model.teacache.nodes import TeaCache
    return TeaCache().apply_teacache(
        model, model_type, rel_l1_thresh, start_percent, end_percent, cache_device
    )[0]


def _apply_magcache(model, model_type, magcache_thresh, retention_ratio, magcache_K, start_step, end_step):
    from .model.magcache.nodes import MagCache
    return MagCache().apply_magcache(
        model, model_type, magcache_thresh, retention_ratio, magcache_K, start_step, end_step
    )[0]

# 定义不同模型对应的参数配置
# node_type:
#   "sampler" -> 没有 model 输入，只输出 SAMPLER
#   "model"   -> 输入 MODEL，输出 MODEL
MODEL_PARAMS = {
    "AdaptiveDiff": {
        "node_type": "sampler",
        "params": {
            "threshold": {"default": 0.01, "min": 0, "max": 1.0, "step": 0.01, "display": "threshold"},
            "max_skip_steps": {"default": 3, "min": 0, "max": 10, "step": 1, "display": "max_skip_steps"},
        },
    },
    "EasyCache": {
        "node_type": "sampler",
        "params": {
            "threshold": {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001, "display": "threshold"},
            "ret_steps": {"default": 5, "min": 0, "max": 100, "step": 1, "display": "ret_steps"},
        },
    },
    "TeaCache": {
        "node_type": "model",
        "params": {
            "teacache_model_type": {"default": "wan2.1_t2v_1.3B_ret_mode", "display": "model_type"},
            "rel_l1_thresh": {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "display": "rel_l1_thresh"},
            "start_percent": {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "start_percent"},
            "end_percent": {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "end_percent"},
            "cache_device": {"default": "cuda", "display": "cache_device"},
        },
    },
    "MagCache": {
        "node_type": "model",
        "params": {
            "magcache_model_type": {"default": "wan2.1_t2v_1.3B", "display": "model_type"},
            "magcache_thresh": {"default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01, "display": "magcache_thresh"},
            "retention_ratio": {"default": 0.2, "min": 0.1, "max": 0.3, "step": 0.01, "display": "retention_ratio"},
            "magcache_K": {"default": 2, "min": 0, "max": 6, "step": 1, "display": "magcache_K"},
            "start_step": {"default": 0, "min": 0, "max": 100, "step": 1, "display": "start_step"},
            "end_step": {"default": -1, "min": -100, "max": 100, "step": 1, "display": "end_step"},
        },
    },
}

# 获取所有可能的参数名
#ALL_PARAM_NAMES = ["param1", "param2", "param3", "param4", "param5", "param6"]


class DynamicModelParamsNode:
    """
    根据选择的模型动态展示不同参数的节点，返回 SAMPLER
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MODEL_PARAMS.keys()), {"default": "AdaptiveDiff"}),
            },
            "optional": {
                # sampler_name 仅 node_type == "sampler" 的模型使用，
                # 前端会根据 model_name 动态显隐
                'sampler_name': (support_sampler,),
                # MODEL 输入：仅 node_type == "model" 的模型需要使用，
                # 前端会根据 model_name 动态显隐对应的 input slot
                "model": ("MODEL",),
                "sampler_name": (support_sampler,),
                "max_skip_steps": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}), #AdaptiveDiff
                "threshold": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001}),
                # threshold 同时被 AdaptiveDiff（默认 0.01）与 EasyCache（默认 0.025）使用，
                # 共享一个 widget；step 取细粒度 0.001 以兼容两者
                "threshold": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001}),
                "ret_steps": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}), #EasyCache
                "param1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "param2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "param3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "param4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "param5": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "param6": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "teacache_model_type": (TEACACHE_MODEL_TYPES, {"default": "wan2.1_t2v_1.3B_ret_mode"}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache_device": (["cuda", "cpu"], {"default": "cuda"}),
                "magcache_model_type": (MAGCACHE_MODEL_TYPES, {"default": "wan2.1_t2v_1.3B"}),
                "magcache_thresh": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01}),
                "retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.3, "step": 0.01}),
                "magcache_K": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "end_step": ("INT", {"default": -1, "min": -100, "max": 100, "step": 1}),
            },
        }

    # 同时声明 SAMPLER 和 MODEL 两个输出，前端根据 model_name 动态显隐对应 output slot；
    # 未使用的输出在 get_sampler 中返回 None
    RETURN_TYPES = ("SAMPLER", "MODEL")
    RETURN_NAMES = ("sampler", "model")
    FUNCTION = "get_sampler"
    CATEGORY = "AccelDiff"
    
    @classmethod
    def IS_CHANGED(cls, model_name, **kwargs):
        return model_name

    def get_sampler(self, model_name,
                    sampler_name="euler",
                    model=None,
                    max_skip_steps=3,                    # AdaptiveDiff 参数
                    threshold=0.025,                     # AdaptiveDiff & EasyCache 共享：阈值
                    ret_steps=5,                         # EasyCache 参数
                    param1=0.5, param2=0.5, param3=0.5,
                    param4=0.5, param5=0.5, param6=0.5,  # 其他模型参数
                    teacache_model_type="wan2.1_t2v_1.3B_ret_mode",
                    rel_l1_thresh=0.4,
                    start_percent=0.0,
                    end_percent=1.0,
                    cache_device="cuda",
                    magcache_model_type="wan2.1_t2v_1.3B",
                    magcache_thresh=0.06,
                    retention_ratio=0.2,
                    magcache_K=2,
                    start_step=0,
                    end_step=-1,
                    **kwargs):

        node_type = MODEL_PARAMS.get(model_name, {}).get("node_type", "sampler")

        # ====== sampler 类节点 ======
        if node_type == "sampler":
            extra_options = {}
            sampler = None
            if sampler_name == "ddim":
                if model_name == 'AdaptiveDiff':
                    from .sampler.adaptivediff import KSAMPLER, sample_euler
                    sampler = KSAMPLER(model_name, sample_euler, max_skip_steps=max_skip_steps, threshold=threshold, extra_options=extra_options, inpaint_options={"random": True})
                elif model_name == 'EasyCache':
                    from .sampler.easycache import KSAMPLER, sample_euler
                    sampler = KSAMPLER(model_name, sample_euler, threshold=threshold, ret_steps=ret_steps, extra_options=extra_options, inpaint_options={"random": True})
            elif sampler_name == "dpm_fast":
                if model_name == 'AdaptiveDiff':
                    from .sampler.adaptivediff import KSAMPLER, sample_dpm_fast
                    def dpm_fast_function(model, noise, sigmas, max_skip_steps=3, threshold=0.01, extra_args=None, callback=None, disable=None, **extra_opts):
                        if len(sigmas) <= 1:
                            return noise
                        sigma_min = sigmas[-1]
                        if sigma_min == 0:
                            sigma_min = sigmas[-2]
                        total_steps = len(sigmas) - 1
                        return sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable, max_skip_steps=max_skip_steps, threshold=threshold)
                    sampler = KSAMPLER(model_name, dpm_fast_function, max_skip_steps=max_skip_steps, threshold=threshold, extra_options=extra_options, inpaint_options={"random": True})
                elif model_name == 'EasyCache':
                    from .sampler.easycache import KSAMPLER, sample_dpm_fast
                    def dpm_fast_function(model, noise, sigmas, threshold=0.025, ret_steps=5, extra_args=None, callback=None, disable=None, **extra_opts):
                        if len(sigmas) <= 1:
                            return noise
                        sigma_min = sigmas[-1]
                        if sigma_min == 0:
                            sigma_min = sigmas[-2]
                        total_steps = len(sigmas) - 1
                        return sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, threshold=threshold, ret_steps=ret_steps, extra_args=extra_args, callback=callback, disable=disable)
                    sampler = KSAMPLER(model_name, dpm_fast_function, threshold=threshold, ret_steps=ret_steps, extra_options=extra_options, inpaint_options={"random": True})
            elif sampler_name == "dpm_adaptive":
                if model_name == 'AdaptiveDiff':
                    from .sampler.adaptivediff import KSAMPLER, sample_dpm_adaptive
                    def dpm_adaptive_function(model, noise, sigmas, max_skip_steps=3, threshold=0.01, extra_args=None, callback=None, disable=None, **extra_opts):
                        if len(sigmas) <= 1:
                            return noise
                        sigma_min = sigmas[-1]
                        if sigma_min == 0:
                            sigma_min = sigmas[-2]
                        total_steps = len(sigmas) - 1
                        return sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable, max_skip_steps=max_skip_steps, threshold=threshold)
                    sampler = KSAMPLER(model_name, dpm_adaptive_function, max_skip_steps=max_skip_steps, threshold=threshold, extra_options=extra_options)
                elif model_name == 'EasyCache':
                    from .sampler.easycache import KSAMPLER, sample_dpm_adaptive
                    def dpm_adaptive_function(model, noise, sigmas, threshold=0.025, ret_steps=5, extra_args=None, callback=None, disable=None, **extra_opts):
                        if len(sigmas) <= 1:
                            return noise
                        sigma_min = sigmas[-1]
                        if sigma_min == 0:
                            sigma_min = sigmas[-2]
                        total_steps = len(sigmas) - 1
                        return sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], total_steps, threshold=threshold, ret_steps=ret_steps, extra_args=extra_args, callback=callback, disable=disable)
                    sampler = KSAMPLER(model_name, dpm_adaptive_function, threshold=threshold, ret_steps=ret_steps, extra_options=extra_options)
            else:
                if model_name == 'AdaptiveDiff':
                    from .sampler.adaptivediff import KSAMPLER
                    from .sampler import adaptivediff as k_diffusion_sampling
                    sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
                    sampler = KSAMPLER(model_name, sampler_function, max_skip_steps=max_skip_steps, threshold=threshold, extra_options=extra_options)
                elif model_name == 'EasyCache':
                    from .sampler.easycache import KSAMPLER
                    from .sampler import easycache as k_diffusion_sampling
                    sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
                    sampler = KSAMPLER(model_name, sampler_function, threshold=threshold, ret_steps=ret_steps, extra_options=extra_options)

            # 该分支只输出 sampler，model 输出位置返回 None
            return (sampler, None)

        # ====== model 类节点 ======
        else:
            # 这里根据 model_name 对传入的 model 做你需要的处理
            # 占位实现：直接把 model 透传出去
            if model is None:
                raise ValueError(f"{model_name} requires a MODEL input")

            if model_name == "TeaCache":
                out_model = _apply_teacache(
                    model,
                    teacache_model_type,
                    rel_l1_thresh,
                    start_percent,
                    end_percent,
                    cache_device,
                )
            elif model_name == "MagCache":
                out_model = _apply_magcache(
                    model,
                    magcache_model_type,
                    magcache_thresh,
                    retention_ratio,
                    magcache_K,
                    start_step,
                    end_step,
                )
            else:
                out_model = model
            # 该分支只输出 model，sampler 输出位置返回 None
            return (None, out_model)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AccelDiffUnified": DynamicModelParamsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AccelDiffUnified": "AccelDiff Unified"
}
