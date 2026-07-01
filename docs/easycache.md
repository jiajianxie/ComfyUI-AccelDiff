# EasyCache

**Category**: Sampler-level acceleration

## Overview

EasyCache introduces a lightweight, runtime-adaptive caching mechanism that dynamically reuses previously computed transformation vectors, avoiding redundant computations during inference. 

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sampler_name` | enum | euler | see supported list | Base sampler algorithm |
| `ec_threshold` | float | 0.025 | 0.0 ~ 1.0 | Accumulated error threshold; lower = higher quality |
| `ret_steps` | int | 5 | 0 ~ 100 | Number of initial steps forced to compute (warmup protection) |

## Parameter Tuning Guide

- **ec_threshold**: Controls the aggressiveness of caching. `0.01~0.03` for high quality, `0.05~0.1` for more speedup. Too high may cause blurriness or artifacts.
- **ret_steps**: Warmup period where no caching is applied (early steps have high variation). Recommended: `3~8` for 20-50 step schedules.

## Official Recommended Parameters by Model

| Model | ec_threshold | ret_steps | Note |
|-------|-----------|-----------|------|
| HunyuanVideo | 0.025 | 5 | Official default, conservative/high-quality |
| Wan2.1 | 0.05 | 10 | Official argparse default, more aggressive |
| Wan2.2 | 0.05 | 10 | Same as Wan2.1 |

> The node defaults (`ec_threshold=0.025`, `ret_steps=5`) follow the HunyuanVideo configuration. For Wan series models, consider increasing both values for better speedup.

## Supported Samplers

`euler`, `heun`, `heunpp2`, `dpm_2`, `lms`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2m`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `gradient_estimation`, `ddim`, `euler_ancestral`, `dpm_2_ancestral`

## Reference

- **Paper**: [*Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching*](https://arxiv.org/abs/2507.02860)
- **GitHub**: [https://github.com/H-EmbodVis/EasyCache](https://github.com/H-EmbodVis/EasyCache)

## Citation

```bibtex
@article{zhou2025easycache,
  title={Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching},
  author={Zhou, Xin and Liang, Dingkang and Chen, Kaijin and Feng, Tianrui and Chen, Xiwu and Lin, Hongkai and Ding, Yikang and Tan, Feiyang and Zhao, Hengshuang and Bai, Xiang},
  journal={arXiv preprint arXiv:2507.02860},
  year={2025}
}
```
