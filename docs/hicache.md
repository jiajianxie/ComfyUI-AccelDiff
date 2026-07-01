# HiCache

**Category**: Model-level acceleration

## Overview

HiCache (Hermite Polynomial-based Feature Cache) is a training-free acceleration framework that improves feature prediction by aligning mathematical tools with empirical properties. The key insight is that feature-derivative approximations in diffusion Transformers exhibit multivariate Gaussian characteristics, motivating the use of Hermite polynomials as a potentially optimal basis for Gaussian-correlated processes. HiCache further introduces a dual-scaling mechanism that ensures numerical stability while preserving predictive accuracy, and is also effective when applied standalone or integrated with TaylorSeer.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `hicache_model_type` | enum | flux | see list below | Target model architecture type |
| `hicache_prediction_mode` | enum | hicache | hicache, taylorseer | Prediction algorithm |
| `hicache_max_order` | int | 1 | 0 ~ 4 | Maximum prediction order |
| `hicache_fresh_threshold` | int | 6 | 2 ~ 20 | Steps between forced full computations |
| `hicache_first_enhance` | int | 3 | 1 ~ 10 | Number of initial warmup steps |
| `hicache_scale_factor` | float | 0.5 | 0.01 ~ 2.0 | Scale factor for hierarchical caching levels |

## Parameter Tuning Guide

- **hicache_prediction_mode**:
  - `hicache`: Uses the hierarchical scale-factor mechanism (recommended)
  - `taylorseer`: Falls back to standard Taylor prediction (useful for comparison)
- **hicache_max_order**: Same as TaylorSeer's `max_order`. `1` is a good default.
- **hicache_fresh_threshold**: Forced recomputation interval. `5~8` is balanced.
- **hicache_first_enhance**: Warmup steps. `2~4` for first-order prediction.
- **hicache_scale_factor**: The key differentiator from TaylorSeer.
  - `0.5`: Default — applies 50% weight to cached predictions (balanced)
  - `0.3~0.5`: More conservative, closer to full computation quality
  - `0.5~1.0`: More aggressive caching, faster but potential quality loss
  - `>1.0`: Amplifies cached predictions (experimental, use with caution)

## Supported Model Types

`flux`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`

## Reference

- **Paper**: [*HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration*](https://arxiv.org/abs/2508.16984)
- **GitHub**: [https://github.com/fenglang918/HiCache](https://github.com/fenglang918/HiCache)

## Citation

```bibtex
@inproceedings{feng2026hicache,
  title={HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration},
  author={Feng, Liang and Zheng, Shikang and Liu, Jiacheng and Lin, Yuqi and Zhou, Qinming and Cai, Peiliang and Wang, Xinyu and Chen, Junjie and Zou, Chang and Ma, Yue and Zhang, Linfeng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
