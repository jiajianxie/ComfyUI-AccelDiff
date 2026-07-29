# TC-Pade

**Category**: Model-level acceleration

## Overview

TC-Pade (Trajectory-Consistent Pade Approximation) is a training-free feature prediction method for diffusion acceleration. It predicts whole-transformer residuals with a Pade-inspired rational approximation instead of directly reusing features or using Taylor-only polynomial extrapolation.

The method uses cached residual history to decide whether the trajectory is stable enough for prediction, then reconstructs the current feature output as:

```text
output ~= current_input + predicted_residual
```

This ComfyUI integration keeps TC-Pade as a model-level method: it takes a `MODEL` input, patches the model forward path, and outputs a patched `MODEL`.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `tcpade_model_type` | enum | wan2.1 | flux, wan2.1 | Target model architecture |
| `tcpade_start_step` | int | 4 | 0 ~ 100 | Step after which prediction can begin |
| `tcpade_end_step` | int | -1 | -1 ~ 200 | Step before which prediction can be used; `-1` means the final sampling step |
| `tcpade_interval` | int | 8 | 1 ~ 100 | Forced full-compute interval; steps divisible by this interval are refreshed |
| `tcpade_n_threshold` | float | 1.4 | 0.0 ~ 5.0 | Trajectory stability threshold used by the curvature test |
| `tcpade_predictor_order` | int | 3 | 1 ~ 4 | Pade predictor order |
| `tcpade_history_size` | int | 6 | 3 ~ 12 | Number of residuals retained in history |
| `tcpade_cache_device` | enum | cuda | cuda, cpu | Device used for residual history storage |

## Supported Model Types

`flux`, `wan2.1`

The official public implementation patches a Diffusers FLUX transformer forward. This ComfyUI adapter maps the same whole-block residual prediction rule to ComfyUI FLUX and Wan2.1 model forwards. The Wan2.1 path follows the paper's model-level residual prediction design, but should be treated as an integration target that requires per-model validation.

## Usage

Use `AccelDiff Unified` with:

- `model_method = TC-Pade`
- Connect your model loader output to the node's `model` input
- Connect the node's `model` output to the sampler/KSampler model input

TC-Pade can be used alone or alongside a sampler-level method through the separate `sampler_method` selector.

## Reference

- **Paper**: *TC-Pade: Trajectory-Consistent Pade Approximation for Diffusion Acceleration*
- **GitHub**: [Alibaba-VELLDEPTH/TC_Pade](https://github.com/Alibaba-VELLDEPTH/TC_Pade)

## Notes

The node logs prediction usage as `[TC-Pade] predicted residual ...` during execution. A successful hook or prediction log only proves that the patch and prediction path executed; it does not prove final visual quality for a given model, prompt, resolution, or sampling schedule.
