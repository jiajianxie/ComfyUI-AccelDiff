# MagCache

**Category**: Model-level acceleration

## Overview

MagCache introduces a novel and robust discovery: a unified magnitude law observed across different models and prompts. Specifically, the magnitude ratio of successive residual outputs decreases monotonically, steadily in most timesteps while rapidly in the last several steps. Leveraging this insight, MagCache adaptively skips unimportant timesteps using an error modeling mechanism and adaptive caching strategy. Unlike existing methods requiring dozens of curated samples for calibration, MagCache only requires a single sample for calibration.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `magcache_model_type` | enum | wan2.1_t2v_1.3B | see list below | Target model architecture type |
| `magcache_thresh` | float | 0.06 | 0.0 ~ 0.3 | Magnitude threshold for cache hit; higher = more skipping |
| `retention_ratio` | float | 0.2 | 0.1 ~ 0.3 | Ratio of layers retained during caching |
| `magcache_K` | int | 2 | 0 ~ 6 | Number of steps to cache before forced re-computation |
| `start_step` | int | 0 | 0 ~ 100 | Step index to start applying MagCache |
| `end_step` | int | -1 | -100 ~ 100 | Step index to stop applying MagCache (-1 = last step) |

## Parameter Tuning Guide

- **magcache_thresh**: Core threshold for deciding cache hits.
  - `0.03~0.06`: Conservative, safe quality
  - `0.06~0.12`: Balanced speedup (~2x)
  - `>0.12`: Aggressive, higher risk of artifacts
- **retention_ratio**: Fraction of transformer layers that always compute fully.
  - `0.2`: Default — retains 20% of layers, skips 80%
  - `0.3`: More conservative (retains 30%)
  - `0.1`: More aggressive
- **magcache_K**: Cache validity period. After K steps, all layers re-compute.
  - `2`: Re-compute every 2 steps (safe, ~1.5x)
  - `3~4`: More aggressive (~2-3x)
- **start_step / end_step**: Limit the active range. Set `end_step=-1` for "until the last step".

## Supported Model Types

`flux`, `flux_kontext`, `chroma`, `qwen_image`, `hunyuan_video`, `hunyuan_video1.5`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`, `wan2.1_i2v_480p_14B`, `wan2.1_i2v_720p_14B`, `wan2.1_vace_1.3B`, `wan2.1_vace_14B`

## Reference

- **Paper**: [*MagCache: Fast Video Generation with Magnitude-Aware Cache*](https://arxiv.org/abs/2506.09045)
- **GitHub**: [https://github.com/Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache)

## Citation

```bibtex
@inproceedings{
  ma2025magcache,
  title={MagCache: Fast Video Generation with Magnitude-Aware Cache},
  author={Zehong Ma and Longhui Wei and Feng Wang and Shiliang Zhang and Qi Tian},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=KZn7TDOL4J}
}
```
