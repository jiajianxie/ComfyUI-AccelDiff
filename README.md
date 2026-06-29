# ComfyUI-AccelDiff

A unified ComfyUI custom node for **training-free diffusion acceleration**. This node integrates multiple acceleration methods — including both **sampler-level** (model output reuse/approximation) and **model-level** (attention/layer feature caching) approaches — into a single, easy-to-use node. Users can select different acceleration strategies without needing a separate node for each method.

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## 📖 Introduction

Diffusion model inference is computationally expensive. Many **training-free** acceleration methods have been proposed to speed up the sampling process without retraining the model. These methods generally fall into two categories:

- **Sampler-level methods**: Reuse or approximate the model's output (predicted noise/denoised result) across timesteps to skip redundant denoising computations.
- **Model-level methods**: Cache and reuse intermediate features (e.g., attention maps, transformer block outputs) inside the model architecture to reduce per-step computation.

**ComfyUI-AccelDiff** unifies these methods into **one single node** (`AccelDiff Unified`), allowing users to:
- Select an acceleration method from a dropdown
- Customize parameters to achieve better speed-quality trade-offs for your specific use case
- Seamlessly integrate acceleration into existing ComfyUI workflows

### Supported Methods

| Method | Category | Description | Reference |
|--------|----------|-------------|-----------|
| **AdaptiveDiff** | Sampler | Adaptively skips denoising steps based on output similarity threshold | [AdaptiveDiff](https://github.com/InternScience/AdaptiveDiffusion) |
| **EasyCache** | Sampler | Caches denoised-input residual and reuses it when predicted error is below threshold | [EasyCache](https://github.com/H-EmbodVis/EasyCache) |
| **TeaCache** | Model | Caches transformer block outputs using relative L1 distance to detect redundant computation | [TeaCache](https://github.com/ali-vilab/TeaCache) |
| **MagCache** | Model | Uses magnitude-based ratios to selectively cache and skip transformer layers | [MagCache](https://github.com/Zehong-Ma/MagCache) |

---

## 🆕 Updates

- **[2025-06]** 🎉 Initial release with support for **AdaptiveDiff**, **EasyCache**, **TeaCache**, and **MagCache**.

---

## 🔧 Installation

### Method 1: ComfyUI Manager (Recommended)

Search for `ComfyUI-AccelDiff` in ComfyUI Manager and install directly.

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
```

2. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/ComfyUI-AccelDiff.git
```

3. Install dependencies:

```bash
cd ComfyUI-AccelDiff
pip install -r requirements.txt
```

4. Restart ComfyUI.

### Requirements

- ComfyUI (latest version recommended)
- Python >= 3.10
- PyTorch >= 2.1.0
- See `requirements.txt` for full dependencies

---

## 🚀 Usage

### Finding the Node

The node is located at: **AccelDiff** → `AccelDiff Unified`

### How It Works

1. **Select an acceleration method** from the `model_name` dropdown.
2. The node UI **dynamically updates** to show only the relevant parameters and I/O slots:
   - **Sampler-type methods** (AdaptiveDiff, EasyCache): Output a `SAMPLER` — connect it to your KSampler node's sampler input.
   - **Model-type methods** (TeaCache, MagCache): Accept a `MODEL` input and output an accelerated `MODEL` — insert it between your model loader and KSampler.
3. **Configure parameters** according to your quality/speed trade-off preferences.

### Workflow Examples

#### Sampler-type (AdaptiveDiff / EasyCache)

```
[Model Loader] → [KSampler (sampler ← Dynamic Model Sampler)]
```

#### Model-type (TeaCache / MagCache)

```
[Model Loader] → [Dynamic Model Sampler] → [KSampler]
```

---

## 📋 Parameters Reference

### AdaptiveDiff (Sampler)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sampler_name` | enum | euler | see list below | Base sampler algorithm |
| `threshold` | float | 0.01 | 0.0 ~ 1.0 | Similarity threshold for skipping steps; lower = higher quality, less acceleration |
| `max_skip_steps` | int | 3 | 0 ~ 10 | Maximum consecutive steps allowed to skip |

### EasyCache (Sampler)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sampler_name` | enum | euler | see list below | Base sampler algorithm |
| `threshold` | float | 0.025 | 0.0 ~ 1.0 | Accumulated error threshold; lower = higher quality |
| `ret_steps` | int | 5 | 0 ~ 100 | Number of initial steps forced to compute (warmup protection) |

### TeaCache (Model)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `teacache_model_type` | enum | wan2.1_t2v_1.3B_ret_mode | see list below | Target model architecture type |
| `rel_l1_thresh` | float | 0.4 | 0.0 ~ 10.0 | Relative L1 threshold for caching decision; higher = more aggressive caching |
| `start_percent` | float | 0.0 | 0.0 ~ 1.0 | Start applying TeaCache from this percentage of total steps |
| `end_percent` | float | 1.0 | 0.0 ~ 1.0 | Stop applying TeaCache at this percentage of total steps |
| `cache_device` | enum | cuda | cuda, cpu | Device to store cached features |

### MagCache (Model)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input diffusion model (required) |
| `magcache_model_type` | enum | wan2.1_t2v_1.3B | see list below | Target model architecture type |
| `magcache_thresh` | float | 0.06 | 0.0 ~ 0.3 | Magnitude threshold for cache hit; higher = more skipping |
| `retention_ratio` | float | 0.2 | 0.1 ~ 0.3 | Ratio of layers retained during caching |
| `magcache_K` | int | 2 | 0 ~ 6 | Number of steps to cache before re-computation |
| `start_step` | int | 0 | 0 ~ 100 | Step index to start applying MagCache |
| `end_step` | int | -1 | -100 ~ 100 | Step index to stop applying MagCache (-1 = last step) |

---

## 📝 Supported Configurations

### Supported Samplers (for Sampler-type methods)

`euler`, `heun`, `heunpp2`, `dpm_2`, `lms`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2m`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `gradient_estimation`, `ddim`, `euler_ancestral`, `dpm_2_ancestral`

### Supported Model Types — TeaCache

`flux`, `flux-kontext`, `ltxv`, `lumina_2`, `hunyuan_video`, `hidream_i1_full`, `hidream_i1_dev`, `hidream_i1_fast`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`, `wan2.1_i2v_480p_14B`, `wan2.1_i2v_720p_14B`, `wan2.1_t2v_1.3B_ret_mode`, `wan2.1_t2v_14B_ret_mode`, `wan2.1_i2v_480p_14B_ret_mode`, `wan2.1_i2v_720p_14B_ret_mode`

### Supported Model Types — MagCache

`flux`, `flux_kontext`, `chroma`, `qwen_image`, `hunyuan_video`, `hunyuan_video1.5`, `wan2.1_t2v_1.3B`, `wan2.1_t2v_14B`, `wan2.1_i2v_480p_14B`, `wan2.1_i2v_720p_14B`, `wan2.1_vace_1.3B`, `wan2.1_vace_14B`

---

## 🏗️ Project Structure

```
ComfyUI-AccelDiff/
├── __init__.py              # Node registration
├── nodes.py                 # Main node logic with dynamic UI
├── js/
│   └── mine.js              # Frontend JS for dynamic widget show/hide
├── sampler/
│   ├── adaptivediff.py      # AdaptiveDiff sampler implementation
│   ├── easycache.py         # EasyCache sampler implementation
│   └── other.py             # Reserved for future sampler methods
├── model/
│   ├── teacache/            # TeaCache model-level acceleration
│   │   ├── nodes.py
│   │   └── models/          # Model-specific TeaCache implementations
│   └── magcache/            # MagCache model-level acceleration
│       ├── nodes.py
│       └── nodes_calibration.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add a new acceleration method:

1. Fork this repository
2. Choose the appropriate category:
   - Sampler-level → add implementation in `sampler/`
   - Model-level → add a new directory in `model/`
3. Register parameters in `MODEL_PARAMS` (in `nodes.py`) and `MODEL_CONFIG` (in `js/mine.js`)
4. Submit a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — The powerful and modular diffusion UI framework
- [TeaCache](https://github.com/ali-vilab/TeaCache) — Training-free acceleration via timestep embedding aware caching
- [MagCache](https://github.com/MagCache/MagCache) — Magnitude-based adaptive caching for diffusion transformers
- [EasyCache](https://github.com/EasyCacheTeam/EasyCache) — Easy-to-use output caching for diffusion sampling

---

## ⭐ Star History

If this project helps your workflow, please consider giving it a ⭐!
