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

#### Sampler-level (step-skipping / output reuse)

| Method | Description | Docs |
|--------|-------------|------|
| **AdaptiveDiff** | Adaptive step-skipping based on output similarity | [📄 Details](docs/adaptivediff.md) |
| **EasyCache** | Cached residual reuse with error threshold | [📄 Details](docs/easycache.md) |
| **SADA** | Momentum-based adaptive skip + Lagrange interpolation (sampler+model) | [📄 Details](docs/sada.md) |
| **ZEUS** | Fixed-pattern modular skip + PSI/Lagrange interpolation | [📄 Details](docs/zeus.md) |

#### Model-level (feature caching / layer skipping)

| Method | Description | Docs |
|--------|-------------|------|
| **TeaCache** | Timestep-embedding-aware transformer output caching | [📄 Details](docs/teacache.md) |
| **MagCache** | Magnitude-based adaptive layer caching | [📄 Details](docs/magcache.md) |
| **TaylorSeer** | Taylor-expansion prediction of transformer outputs | [📄 Details](docs/taylorseer.md) |
| **HiCache** | Hierarchical caching with multi-order prediction | [📄 Details](docs/hicache.md) |
| **SeaCache** | Similarity-based exponential adaptive caching | [📄 Details](docs/seacache.md) |

---

## 🆕 Updates

- **[2025-06]** Added **SADA**, **ZEUS**, **TaylorSeer**, **HiCache**, and **SeaCache** methods.
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

1. **Select an acceleration method** from the dropdown.
2. The node UI **dynamically updates** to show only the relevant parameters and I/O slots:
   - **Sampler-type methods** (AdaptiveDiff, EasyCache, SADA, ZEUS): Output a `SAMPLER` — connect it to your KSampler node's sampler input.
   - **Model-type methods** (TeaCache, MagCache, TaylorSeer, HiCache, SeaCache): Accept a `MODEL` input and output an accelerated `MODEL` — insert it between your model loader and KSampler.
   - **SADA** requires both sampler + model to work together (outputs both).
3. **Configure parameters** according to your quality/speed trade-off preferences.

### Workflow Examples

#### Sampler-type (AdaptiveDiff / EasyCache / ZEUS)

```
[Model Loader] → [KSampler (sampler ← AccelDiff Unified)]
```

#### Model-type (TeaCache / MagCache / TaylorSeer / HiCache / SeaCache)

```
[Model Loader] → [AccelDiff Unified] → [KSampler]
```

#### SADA (Sampler + Model combined)

```
[Model Loader] → [AccelDiff Unified] → [KSampler (sampler ← AccelDiff Unified, model ← AccelDiff Unified)]
```

---

## 📋 Parameters Reference

Each method has its own detailed documentation with full parameter tables, tuning guides, and citations. Click the links below:

#### Sampler Methods

- [AdaptiveDiff](docs/adaptivediff.md) — Adaptive step-skipping
- [EasyCache](docs/easycache.md) — Error-threshold residual caching
- [SADA](docs/sada.md) — Momentum-based skip + Lagrange interpolation
- [ZEUS](docs/zeus.md) — Fixed-pattern modular skip + multi-strategy interpolation

#### Model Methods

- [TeaCache](docs/teacache.md) — Timestep-embedding-aware caching
- [MagCache](docs/magcache.md) — Magnitude-based layer caching
- [TaylorSeer](docs/taylorseer.md) — Taylor-expansion prediction
- [HiCache](docs/hicache.md) — Hierarchical multi-order caching
- [SeaCache](docs/seacache.md) — Similarity-based exponential adaptive caching

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
│   ├── sada.py              # SADA sampler implementation
│   └── zeus.py              # ZEUS sampler implementation
├── model/
│   ├── teacache/            # TeaCache model-level acceleration
│   ├── magcache/            # MagCache model-level acceleration
│   ├── taylorseer/          # TaylorSeer model-level acceleration
│   ├── hicache/             # HiCache model-level acceleration
│   └── seacache/            # SeaCache model-level acceleration
├── docs/                    # Per-method detailed documentation
│   ├── adaptivediff.md
│   ├── easycache.md
│   ├── sada.md
│   ├── zeus.md
│   ├── teacache.md
│   ├── magcache.md
│   ├── taylorseer.md
│   ├── hicache.md
│   └── seacache.md
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
- [AdaptiveDiff](https://github.com/InternScience/AdaptiveDiffusion) — Adaptive step-skipping based on output similarity
- [EasyCache](https://github.com/EasyCacheTeam/EasyCache) — Easy-to-use output caching for diffusion sampling
- [SADA](https://github.com/SADA-Diffusion/SADA) — Step-adaptive acceleration with momentum-based decision
- [ZEUS](https://github.com/ZEUS-Diffusion/ZEUS) — Zero-shot efficient unified sparsity for diffusion acceleration
- [TeaCache](https://github.com/ali-vilab/TeaCache) — Training-free acceleration via timestep embedding aware caching
- [MagCache](https://github.com/Zehong-Ma/MagCache) — Magnitude-based adaptive caching for diffusion transformers
- [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) — Taylor-expansion based prediction for transformer caching
- [HiCache](https://github.com/HiCache-Diffusion/HiCache) — Hierarchical caching with multi-order prediction
- [SeaCache](https://github.com/SeaCache/SeaCache) — Similarity-based exponential adaptive caching

---

## ⭐ Star History

If this project helps your workflow, please consider giving it a ⭐!
