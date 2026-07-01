# SADA

**Category**: Sampler-level acceleration (pure sampler method, no model patching)

## Overview

SADA (Stability-guided Adaptive Diffusion Acceleration) is a novel paradigm that unifies step-wise and token-wise sparsity decisions via a single stability criterion to accelerate sampling of ODE-based generative models (Diffusion and Flow-matching). It adaptively allocates sparsity based on the sampling trajectory, and introduces principled approximation schemes that leverage precise gradient information from the numerical ODE solver.

The original paper proposes both step-level acceleration (adaptive step skipping) and token-level sparsity (token pruning). For generality and compatibility with ComfyUI's unified k-diffusion interface, **we only integrate the step-level sampler acceleration** — the core momentum-based stability criterion and Lagrange interpolation reconstruction. This allows SADA to work as a pure sampler method with any model type without requiring model patching or architecture-specific modifications.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `acc_start` | int | 10 | 0 ~ 100 | Step index to start acceleration (min 3) |
| `acc_end` | int | 47 | -100 ~ 200 | Step index to stop acceleration (negative = N+value) |
| `max_interval` | int | 4 | 1 ~ 20 | Maximum consecutive steps to skip |
| `lagrange_term` | int | 3 | 0 ~ 5 | Lagrange interpolation order (0 = disabled) |
| `lagrange_int` | int | 4 | 1 ~ 20 | Lagrange anchor point interval |
| `lagrange_step` | int | 20 | 5 ~ 100 | Step after which Lagrange interpolation activates |

## Parameter Tuning Guide

- **acc_start / acc_end**: Defines the active acceleration window. Early and final steps are typically not skipped (high-change regions). For 50-step schedules, `(10, 47)` is a good default.
- **max_interval**: Limits consecutive skips. `3~5` is safe; higher values risk quality drop.
- **lagrange_term**: Set to `3~4` for higher-quality interpolation at skipped steps. `0` disables Lagrange (uses simple x0 prediction).
- **lagrange_int**: Interval between anchor points for Lagrange interpolation. `4~6` is typical.
- **lagrange_step**: Lagrange only activates after this step (early steps need full computation). `20~30` for 50-step schedules. Must satisfy `lagrange_step % lagrange_int == 0`.

## Supported Samplers

`euler`, `heun`, `heunpp2`, `dpm_2`, `lms`, `dpmpp_2m`, `ddim`, `euler_ancestral`, `dpm_2_ancestral`

## Reference

- **Paper**: [*SADA: Stability-guided Adaptive Diffusion Acceleration*](https://arxiv.org/abs/2507.17135)
- **GitHub**: [https://github.com/Ting-Justin-Jiang/sada-icml](https://github.com/Ting-Justin-Jiang/sada-icml)

## Citation

```bibtex
@inproceedings{jiang2025sada,
  title     = {SADA: Stability-guided Adaptive Diffusion Acceleration},
  author    = {Ting Jiang and Yixiao Wang and Hancheng Ye and Zishan Shao and Jingwei Sun and Jingyang Zhang and Zekai Chen and Jianyi Zhang and Yiran Chen and Hai Li},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025}
}
```
