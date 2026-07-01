# SeaCache

**Category**: Model-level acceleration

## Overview

SeaCache (Spectral-Evolution-Aware Cache) is a training-free cache schedule that bases reuse decisions on a spectrally aligned representation. Through theoretical and empirical analysis, SeaCache derives a Spectral-Evolution-Aware (SEA) filter that preserves content-relevant components while suppressing noise. Employing SEA-filtered input features to estimate redundancy leads to dynamic schedules that adapt to content while respecting the spectral priors underlying the diffusion model.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `seacache_model_type` | enum | flux | see list below | Target model architecture type |
| `seacache_thresh` | float | 0.3 | 0.0 ~ 2.0 | Base similarity threshold for cache decision |
| `seacache_power_exp` | float | 2.0 | 1.0 ~ 5.0 | Power exponent for threshold scheduling |
| `seacache_ret_steps` | int | 1 | 0 ~ 10 | Number of initial steps to always compute (warmup) |

## Parameter Tuning Guide

- **seacache_thresh**: Base threshold for cache hit decision.
  - `0.1~0.3`: Conservative, high quality
  - `0.3~0.5`: Balanced (~2x speedup)
  - `0.5~1.0`: Aggressive (~2-3x speedup)
  - `>1.0`: Very aggressive, potential artifacts
- **seacache_power_exp**: Controls how the threshold changes across steps.
  - `1.0`: Flat (same threshold throughout) — simplest
  - `2.0`: Quadratic ramp — more caching in later steps (recommended)
  - `3.0~5.0`: Steep ramp — very conservative early, very aggressive late
  - Higher values make the method more adaptive to the denoising trajectory shape
- **seacache_ret_steps**: Warmup period. First N steps always compute fully.
  - `1~2`: Minimal warmup (for well-behaved models like FLUX)
  - `3~5`: Safer for complex models or video generation

## Supported Model Types

`flux`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`, `wan2.1_i2v_480p_14B`, `wan2.1_i2v_720p_14B`

## Reference

- **Paper**: [*SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models*](https://arxiv.org/abs/2602.18993)
- **GitHub**: [https://github.com/jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache)

## Citation

```bibtex
@inproceedings{chung2026seacache,
  title={SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models},
  author={Chung, Jiwoo and Hyun, Sangeek and Lee, MinKyu and Han, Byeongju and Cha, Geonho and Wee, Dongyoon and Hong, Youngjun and Heo, Jae-Pil},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14283--14294},
  year={2026}
}
```
