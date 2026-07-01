# TeaCache

**Category**: Model-level acceleration

## Overview

TeaCache (Timestep Embedding Aware Cache) is a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps. Rather than directly using the time-consuming model outputs, TeaCache focuses on model inputs, which have a strong correlation with the model outputs while incurring negligible computational cost. TeaCache first modulates the noisy inputs using the timestep embeddings to ensure their differences better approximating those of model outputs. TeaCache then introduces a rescaling strategy to refine the estimated differences and utilizes them to indicate output caching.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `teacache_model_type` | enum | wan2.1_t2v_1.3B_ret_mode | see list below | Target model architecture type |
| `rel_l1_thresh` | float | 0.4 | 0.0 ~ 10.0 | Relative L1 threshold for caching decision; higher = more aggressive |
| `start_percent` | float | 0.0 | 0.0 ~ 1.0 | Start applying TeaCache from this percentage of total steps |
| `end_percent` | float | 1.0 | 0.0 ~ 1.0 | Stop applying TeaCache at this percentage of total steps |
| `cache_device` | enum | cuda | cuda, cpu | Device to store cached features |

## Parameter Tuning Guide

- **rel_l1_thresh**: The core quality-speed knob.
  - `0.1~0.3`: Conservative, high quality, mild speedup (~1.5x)
  - `0.3~0.6`: Balanced (recommended starting point)
  - `0.6~1.0`: Aggressive, significant speedup (~2-3x), potential quality loss
  - `>1.0`: Very aggressive, may cause noticeable artifacts
- **start_percent / end_percent**: Narrow the active range. Often the first 10-20% and last 10% of steps benefit from full computation. Try `(0.1, 0.9)`.
- **cache_device**: Use `cuda` for fastest access. Switch to `cpu` if GPU memory is tight (video generation with large models).
- **teacache_model_type**: Must match your model. The `_ret_mode` variants use retention-based caching (better for some architectures).

## Supported Model Types

`flux`, `flux-kontext`, `ltxv`, `lumina_2`, `hunyuan_video`, `hidream_i1_full`, `hidream_i1_dev`, `hidream_i1_fast`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`, `wan2.1_i2v_480p_14B`, `wan2.1_i2v_720p_14B`, `wan2.1_t2v_1.3B_ret_mode`, `wan2.1_t2v_14B_ret_mode`, `wan2.1_i2v_480p_14B_ret_mode`, `wan2.1_i2v_720p_14B_ret_mode`

## Reference

- **Paper**: [*Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model*](https://arxiv.org/abs/2411.19108)
- **GitHub**: [https://github.com/ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)

## Citation

```bibtex
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```
