# AdaptiveDiff

**Category**: Sampler-level acceleration

## Overview

AdaptiveDiff aims to skip as many noise prediction steps as possible while keeping the final denoised results identical to the original full-step ones. The skipping strategy is guided by the **third-order latent difference** that indicates the stability between timesteps during the denoising process, enabling the reuse of previous noise prediction results.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sampler_name` | enum | euler | see supported list | Base sampler algorithm |
| `threshold` | float | 0.01 | 0.0 ~ 1.0 | Similarity threshold for skipping steps; lower = higher quality, less acceleration |
| `max_skip_steps` | int | 4 | 0 ~ 10 | Maximum consecutive steps allowed to skip |

## Parameter Tuning Guide

- **threshold**: Start with `0.01` for high quality. Increase to `0.05~0.1` for more aggressive acceleration (2-3x speedup). Values above `0.2` may introduce visible artifacts.
- **max_skip_steps**: Limits how many consecutive steps can be skipped. Higher values allow more acceleration but risk quality degradation in rapid-change regions. Recommended: `2~5`.

## Supported Samplers

`euler`, `heun`, `heunpp2`, `dpm_2`, `lms`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2m`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `gradient_estimation`, `ddim`, `euler_ancestral`, `dpm_2_ancestral`

## Reference

- **Paper**: [*Training-Free Adaptive Diffusion with Bounded Difference Approximation Strategy*](https://arxiv.org/abs/2410.09873)
- **GitHub**: [https://github.com/InternScience/AdaptiveDiffusion](https://github.com/InternScience/AdaptiveDiffusion)

## Citation

```bibtex
@misc{adaptivediffusion24ye,
  author = {Hancheng Ye and Jiakang Yuan and Renqiu Xia and Xiangchao Yan and Tao Chen and Junchi Yan and Botian Shi and Bo Zhang},
  title = {Training-Free Adaptive Diffusion with Bounded Difference Approximation Strategy},
  year = {2024},
  booktitle = {The Thirty-Eighth Annual Conference on Neural Information Processing Systems}
}
```
