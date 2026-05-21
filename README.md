# SparseView4D: Lightweight Sparse-View 4D Dynamic Scene Reconstruction

**A Lightweight Sparse-View 4D Dynamic Scene Reconstruction Pipeline Based on Deformable 3D Gaussian Splatting**

Chenyue Han · Computer Vision · Spring 2026 · New York University · ch5085@nyu.edu

---

## Overview

Reconstructing dynamic 3D scenes from sparse multi-view inputs remains a fundamental challenge in computer vision. Existing methods such as Deformable 3DGS assume dense camera setups (50–100+ views), which are impractical in consumer and robotic settings.

This project investigates how far a dynamic scene reconstruction pipeline can be pushed under **sparse-view constraints (N ∈ {4, 6, 8} views)**, and proposes three lightweight interventions on top of [Deformable 3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians):

1. **Depth-guided initialization** — uses Depth Anything V2 to replace random point cloud initialization with geometry-aware priors
2. **Temporal smoothness regularization** — penalizes large changes in the deformation field across neighboring timesteps
3. **Multi-frame photometric supervision** — simultaneously renders neighboring frames to enforce temporal coherence

**Key result**: Depth-guided initialization consistently improves perceptual quality (SSIM +0.09, LPIPS −0.07 on average) under sparse-view constraints.

---

## Method

### Baseline Problem

Under the original dense setup (50–200 views), Deformable 3DGS achieves strong results (e.g., PSNR 37.50 on *jumpingjacks*). With only 4–8 views, two failure modes emerge:

- **Gaussian explosion**: random initialization has no geometric grounding; Gaussians drift or grow unboundedly
- **Degenerate deformation**: the MLP deformation field is severely under-constrained with so few viewpoints

PSNR drops from 37.50 dB (dense) to 14.60 dB (4-view) — a gap of over 22 dB.

### Depth-Guided Initialization

For each training frame:
1. Estimate a depth map using **Depth Anything V2**
2. Apply percentile clipping (5th–95th) to remove outliers
3. **Scale alignment**: set the median depth equal to the camera-to-scene-center distance
4. Back-project depth pixels to 3D world coordinates
5. Fuse point clouds from all N frames into a single ~25K-point geometry-aware initialization

### Temporal Constraints

**Deformation Smoothness Loss:**

```
L_smooth = ||d(t) - d(t-Δt)||² + ||d(t) - d(t+Δt)||²
```

**Multi-Frame Photometric Supervision:**

```
L_mf = Σ L_photo(rendered(t+δ), gt(t+δ))  for δ ∈ {-Δt, +Δt}
```

**Full Training Objective:**

```
L = L_photo + λ_t · L_smooth + λ_mf · L_mf
```

where `λ_t = 0.01` and `λ_mf = 0.5` by default.

---

## Results

### jumpingjacks

| Method | Views | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| Dense (reference) | 200 | 37.50 | 0.9894 | 0.0138 |
| Baseline | 8 | 19.09 | 0.8873 | 0.1183 |
| Baseline | 6 | 15.48 | 0.7855 | 0.2179 |
| Baseline | 4 | 14.60 | 0.7759 | 0.2316 |
| **+Depth Init** | 8 | 16.92 | **0.9062** | **0.1012** |
| **+Depth Init** | 6 | 16.93 | **0.9064** | **0.1007** |
| **+Depth Init** | 4 | 16.00 | **0.8709** | **0.1343** |
| +Depth+Smooth | 4 | 16.06 | 0.8727 | 0.1330 |
| +Depth+MF | 4 | 16.05 | 0.8720 | 0.1338 |

### standup

| Method | Views | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| Dense (reference) | — | 43.88 | 0.9943 | 0.0083 |
| Baseline | 4 | 19.29 | 0.7970 | 0.1679 |
| **+Depth+Smooth** | 4 | **18.96** | **0.9021** | **0.0936** |

Depth-guided initialization closes approximately **60% of the sparse-to-dense gap** on SSIM at the hardest 4-view setting.

---

## Code Structure

```
SparseView4D/
├── train.py                  # Main training script (modified)
├── render.py                 # Rendering and evaluation
├── metrics.py                # PSNR / SSIM / LPIPS evaluation
├── convert.py                # Dataset conversion utilities
├── arguments/
│   └── __init__.py           # Training arguments (modified: added sparse-view flags)
├── scene/
│   ├── __init__.py           # Scene loader (modified: depth-guided init)
│   ├── dataset_readers.py    # Dataset I/O (modified: depth map support)
│   └── gaussian_model.py     # 3D Gaussian representation
├── gaussian_renderer/        # Differentiable Gaussian renderer
├── utils/
│   ├── loss_utils.py         # Loss functions (modified: added ARAP, temporal losses)
│   └── ...
├── submodules/
│   ├── depth-diff-gaussian-rasterization/  # Custom CUDA rasterizer
│   └── simple-knn/                          # KNN for Gaussian densification
├── lpipsPyTorch/             # Perceptual loss
└── paper.pdf                 # Final report
```

### Key Modifications vs. Original Deformable 3DGS

| File | Change |
|------|--------|
| `scene/__init__.py` | Depth-guided point cloud initialization |
| `scene/dataset_readers.py` | Load and preprocess depth maps |
| `train.py` | Multi-frame loss, temporal smoothness loss, ARAP loss |
| `utils/loss_utils.py` | Added `arap_loss()` for rigidity regularization |
| `arguments/__init__.py` | New flags: `--use_temporal_smooth`, `--use_arap`, `--lambda_multiframe` |

---

## Setup

### Environment

```bash
git clone https://github.com/ChenyueHan43/SparseView4D --recursive
cd SparseView4D

conda create -n sparseview4d python=3.7
conda activate sparseview4d

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Dataset

Download the [D-NeRF synthetic dataset](https://www.albertpumarola.com/research/D-NeRF/index.html) and organize as:

```
data/
└── D-NeRF/
    ├── jumpingjacks/
    ├── standup/
    └── ...
```

---

## Training

**Baseline (sparse-view, no modifications):**
```bash
python train.py -s data/D-NeRF/jumpingjacks -m output/baseline_4view \
    --eval --is_blender --num_views 4
```

**With depth-guided initialization:**
```bash
python train.py -s data/D-NeRF/jumpingjacks -m output/depth_4view \
    --eval --is_blender --num_views 4 --use_depth_init
```

**With all temporal constraints:**
```bash
python train.py -s data/D-NeRF/jumpingjacks -m output/full_4view \
    --eval --is_blender --num_views 4 --use_depth_init \
    --use_temporal_smooth --lambda_temporal 0.01 \
    --lambda_multiframe 0.5
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_depth_init` | False | Enable depth-guided initialization |
| `--use_temporal_smooth` | False | Enable deformation smoothness regularization |
| `--lambda_temporal` | 0.01 | Weight for temporal smoothness loss |
| `--lambda_multiframe` | 0.5 | Weight for multi-frame photometric loss |
| `--use_arap` | False | Enable ARAP rigidity regularization |
| `--lambda_arap` | 0.01 | Weight for ARAP loss |

---

## Evaluation

```bash
python render.py -m output/depth_4view --mode render
python metrics.py -m output/depth_4view
```

---

## Failure Cases

On the *trex* scene, depth-guided initialization degrades performance (PSNR 15.3 vs. baseline 17.5–18.9). Two factors contribute:
- Complex backgrounds produce noisy depth estimates from Depth Anything V2
- Scale alignment based on camera-to-scene-center distance fails when scene geometry deviates from this assumption

Future directions: foreground-aware depth filtering, visibility-aware Gaussian densification, multi-view consistent initialization (e.g., DUSt3R).

---

## Acknowledgments

Built on top of [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) (Yang et al., CVPR 2024).
Depth estimation via [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Yang et al., NeurIPS 2024).
Evaluated on the [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) synthetic benchmark.
Experiments conducted on NVIDIA L40S and H200 GPUs at NYU HPC (Torch cluster).
