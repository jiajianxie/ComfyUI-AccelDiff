# ZEUS

**Category**: Sampler-level acceleration

## Overview

ZEUS is an acceleration method that predicts reduced denoiser evaluations using a second-order predictor, and stabilizes aggressive consecutive skipping with an interleaved scheme that avoids back-to-back extrapolations. ZEUS adds essentially zero overhead, no feature caches, and no architectural modifications, and it is compatible with different backbones, prediction objectives, and solver choices.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `zeus_denominator` | int | 3 | 2 ~ 6 | Sparsity denominator |
| `zeus_modular` | string | "0,1" | - | Comma-separated list of remainders to skip |
| `zeus_acc_start` | int | 10 | 0 ~ 100 | Acceleration window start step |
| `zeus_acc_end` | int | 45 | -100 ~ 200 | Acceleration window end step |
| `zeus_interp_mode` | enum | psi | psi, x_0 | Interpolation mode |
| `zeus_caching_mode` | enum | reuse_interp | reuse_interp, interp_all, reuse_all | Caching strategy |
| `zeus_max_interval` | int | 4 | 1 ~ 20 | Maximum consecutive skipped steps |
| `zeus_lagrange_term` | int | 3 | 0 ~ 5 | Lagrange interpolation order (0 = disabled) |
| `zeus_lagrange_int` | int | 6 | 1 ~ 20 | Lagrange anchor sampling interval |
| `zeus_lagrange_step` | int | 24 | 5 ~ 100 | Step at which Lagrange interpolation activates |

> Default values are from the official `flux_demo.py`.

## Parameter Details

### zeus_denominator (Sparsity Denominator)

Controls the skip period. Every `denominator` steps form a group; within each group, `modular` determines which steps are skipped.

- Larger values → more steps per group → wider range of possible skip ratios
- Typically set to `3` (combined with `modular="0,1"` for a 2/3 skip rate)

### zeus_modular (Skip Remainder List)

Comma-separated list of integers. A step is skipped when `step_index % denominator` matches any value in this list.

- `denominator=3, modular="0,1"` → skip 2 out of every 3 steps, compute only 1/3 → **~3x speedup**
- `denominator=2, modular="0"` → skip 1 out of every 2 steps, compute 1/2 → **~2x speedup**
- `denominator=3, modular="1"` → skip 1 out of every 3 steps, compute 2/3 → **~1.5x speedup**
- `denominator=4, modular="0,1,2"` → skip 3 out of every 4 steps → **~4x speedup** (aggressive)

**Note**: Each value in modular must be < denominator.

### zeus_acc_start / zeus_acc_end (Acceleration Window)

Only steps within the `[acc_start, acc_end)` range are eligible for skipping. Steps outside this window are always fully computed.

- The first few steps (typically 3~10) must be fully computed to establish the initial trajectory
- The last few steps must be fully computed to ensure final output quality
- Internally, `acc_start` is clamped to `max(acc_start, 3)` (the algorithm depends on prior data)
- For 50-step sampling, a typical range is `(8~10, 45~47)`

### zeus_interp_mode (Interpolation Mode)

Determines how model outputs are reconstructed for skipped steps:

- **`psi`** (PSI linear extrapolation): Linear extrapolation in model output space (`epsilon_{t-1} = epsilon_t + (epsilon_t - epsilon_{t+1})`). Fast and suitable for most scenarios. **Recommended as default.**
- **`x_0`** (trajectory interpolation): Uses Simpson's rule in the denoising trajectory space. May be more accurate for complex generation scenarios, but slightly slower.

### zeus_caching_mode (Caching Strategy)

Controls how cached interpolation results are reused during consecutive skipped steps:

- **`reuse_interp`** (recommended): Alternates between new interpolation and cached interpolation during consecutive skips. Best balance between speed and quality.
  - Odd consecutive skips: reuse the previously cached interpolation
  - Even consecutive skips: compute a new interpolation
- **`interp_all`**: Recompute interpolation for every skipped step. Highest quality but more computation.
- **`reuse_all`**: Always reuse the previously cached output. Fastest but lowest quality.

### zeus_max_interval (Maximum Consecutive Skips)

Safety valve — after this many consecutive skipped steps, a full computation is forced.

- Prevents error accumulation from too many consecutive skips
- Typical values: `4` (FLUX/SDXL) or `6` (video models/SD)
- Lower values are more conservative (better quality but reduced speedup)

### zeus_lagrange_term (Lagrange Interpolation Order)

Enables Lagrange polynomial interpolation for higher-precision reconstruction in later steps:

- **`0`**: Disable Lagrange interpolation; use only PSI/x_0
- **`3`**: 3rd-order Lagrange polynomial (recommended for FLUX, CogVideo)
- **`4`**: 4th-order Lagrange polynomial (recommended for SD, SDXL, Wan2.1)

When `lagrange_term > 0`, the algorithm uses PSI/x_0 interpolation in `[acc_start, lagrange_step)` and switches to Lagrange interpolation in `[lagrange_step, acc_end)`.

### zeus_lagrange_int (Lagrange Anchor Interval)

How many steps apart each Lagrange anchor point is recorded (used for polynomial fitting).

- Smaller values → denser anchors → more precise fitting but increased overhead
- Typical values: `6` (FLUX, CogVideo) or `4` (SD, SDXL, Wan2.1)
- **Requirement**: `lagrange_step % lagrange_int == 0`

### zeus_lagrange_step (Lagrange Activation Step)

The step at which Lagrange interpolation replaces PSI/x_0.

- Before this step: PSI/x_0 interpolation is used
- After this step: Lagrange polynomial interpolation is used
- Typical value: `24` (for 50-step sampling)
- **Requirement**: `lagrange_step % lagrange_int == 0`

## Official Recommended Configurations

The following parameters are from the [ZEUS official repository](https://github.com/Ting-Justin-Jiang/ZEUS) demo files, based on 50-step sampling:

### Image Models

| Model | Demo File | denominator | modular | acc_start | acc_end | interp_mode | caching_mode | max_interval | lagrange_term | lagrange_int | lagrange_step |
|-------|-----------|:-----------:|:-------:|:---------:|:-------:|:-----------:|:------------:|:------------:|:-------------:|:------------:|:-------------:|
| **FLUX** | flux_demo.py | 3 | (0,1) | 10 | 45 | psi | reuse_interp | 4 | 3 | 6 | 24 |
| **SD 2.1** | sd_demo.py | 3 | (0,1) | 10 | 45 | psi | reuse_interp | 6 | 4 | 4 | 24 |
| **SDXL** | xl_demo.py | 3 | (0,1) | 10 | 45 | psi | reuse_interp | 4 | 4 | 4 | 24 |

### Video Models

| Model | Demo File | denominator | modular | acc_start | acc_end | interp_mode | caching_mode | max_interval | lagrange_term | lagrange_int | lagrange_step |
|-------|-----------|:-----------:|:-------:|:---------:|:-------:|:-----------:|:------------:|:------------:|:-------------:|:------------:|:-------------:|
| **Wan2.1** | wan2_demo.py | 3 | (0,1) | 8 | 47 | psi | - | 6 | 4 | 4 | 24 |
| **CogVideoX** | cogvideo_demo.py | 3 | (0,1) | 8 | 47 | psi | - | 6 | 3 | 6 | 24 |

### Key Takeaways

1. **All models** use `denominator=3, modular=(0,1)` as the universal configuration (2/3 skip rate, ~3x speedup)
2. **Image models** typically use `acc_range=(10, 45)`, while **video models** use a wider window `acc_range=(8, 47)`
3. **Video models** set `max_interval` to `6` (more permissive), while image models use `4`
4. **SD/SDXL/Wan2.1** prefer `lagrange_term=4, lagrange_int=4` (higher order, denser anchors)
5. **FLUX/CogVideo** prefer `lagrange_term=3, lagrange_int=6` (slightly lower order, sparser anchors)
6. All models use `lagrange_step=24` consistently

## Supported Samplers

`euler`, `heun`, `heunpp2`, `dpm_2`, `lms`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2m`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `gradient_estimation`, `ddim`, `euler_ancestral`, `dpm_2_ancestral`

## Reference

- **Paper**: [*ZEUS: Zero-shot Efficient Unified Sparsity for Diffusion Models*](https://arxiv.org/abs/2604.01552)
- **GitHub**: [https://github.com/Ting-Justin-Jiang/ZEUS](https://github.com/Ting-Justin-Jiang/ZEUS)

## Citation

```bibtex
@misc{zeus2025,
  title        = {ZEUS: Zero-shot Efficient Unified Sparsity for Generative Models},
  author       = {Yixiao Wang and Ting Jiang and Zishan Shao and Hancheng Ye and Jingwei Sun and Mingyuan Ma and Jianyi Zhang and Yiran Chen and Hai Li},
  year         = {2025},
  howpublished = {https://yixiao-wang-stats.github.io/zeus/},
  note         = {Code and project page available at {https://github.com/Ting-Justin-Jiang/ZEUS}}
}
```
