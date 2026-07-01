# TaylorSeer

**Category**: Model-level acceleration

## Overview

TaylorSeer shows that features of diffusion models at future timesteps can be predicted based on their values at previous timesteps. Based on the fact that features change slowly and continuously across timesteps, TaylorSeer employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `taylorseer_model_type` | enum | flux | see list below | Target model architecture type |
| `max_order` | int | 1 | 0 ~ 4 | Maximum Taylor expansion order for prediction |
| `fresh_threshold` | int | 6 | 2 ~ 20 | Steps between forced full computations |
| `first_enhance` | int | 3 | 1 ~ 10 | Number of initial steps with full computation (warmup) |

## Parameter Tuning Guide

- **max_order**: Taylor expansion order.
  - `0`: Zero-th order (simple reuse of last output) — fastest, lowest quality
  - `1`: First-order (linear prediction) — good balance (recommended)
  - `2~3`: Higher-order — better accuracy but needs more history, diminishing returns
  - `4`: Rarely needed, may introduce instability
- **fresh_threshold**: How many steps can use Taylor prediction before a forced full computation.
  - `4~6`: Safe, ensures periodic recalibration
  - `8~12`: More aggressive caching
  - Lower values = more frequent full computation = better quality but less speedup
- **first_enhance**: Warmup period with full computation (Taylor needs history).
  - `2~3`: Minimum warmup for first-order
  - `4~5`: Needed for higher-order Taylor (needs more history points)

## Supported Model Types

`flux`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`, `wan2.1_i2v_480p_14B`, `wan2.1_i2v_720p_14B`

## Reference

- **Paper**: [*From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers*](https://arxiv.org/abs/2503.06923)
- **GitHub**: [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)

## Citation

```bibtex
@article{TaylorSeer2025,
  title={From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers},
  author={Liu, Jiacheng and Zou, Chang and Lyu, Yuanhuiyi and Chen, Junjie and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2503.06923},
  year={2025}
}
```
