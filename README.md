# PS12 · ISRO BAH 2026
## INSAT-3DS Temporal Super-Resolution Dashboard

> **Problem Statement 12** — Enhance the temporal resolution of INSAT-3DS TIR1 imagery from 30 min to 15 min (and optionally 7.5 min) using deep-learning-based optical flow frame interpolation.

---

## Project structure

```
ps12-dashboard/
├── frontend/
│   └── index.html          ← Self-contained dashboard (open in browser)
├── backend/
│   ├── train.py            ← Training script
│   ├── dataset.py          ← GOES-19 & INSAT-3DS data loaders
│   ├── infer.py            ← Inference: generate synthetic .h5 frames
│   └── metrics.py          ← SSIM, PSNR, MSE, FSIM, MCE
├── model/
│   └── model.py            ← Two-network architecture (FlowEstimator + FrameSynthesizer)
├── config.yaml             ← All hyperparameters and paths
└── requirements.txt        ← Python dependencies
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get training data (GOES-19)
```bash
# AWS S3 bucket — no credentials needed (public)
aws s3 sync s3://noaa-goes19/ABI-L1b-RadC/ /data/GOES19/ABI-L1b-RadC/ \
    --no-sign-request \
    --exclude "*" \
    --include "OR_ABI-L1b-RadC-M6C13_G19_s2024*"
```

### 3. Get inference data (INSAT-3DS)
```
Register at: https://mosdac.gov.in
Download:    Products → INSAT-3DS → L1B → TIR1 → 3SIMG_*_TIR1*.h5
Place in:    /data/INSAT3DS/TIR1/
```

### 4. Configure paths
Edit `config.yaml`:
```yaml
data:
  train_root: /data/GOES19/ABI-L1b-RadC/
  insat_root: /data/INSAT3DS/TIR1/
```

### 5. Train
```bash
python backend/train.py --config config.yaml
# Checkpoints → ./checkpoints/best_model.pth
# Logs        → ./logs/  (view with: tensorboard --logdir logs)
```

### 6. Run inference on INSAT-3DS
```bash
# Generate 15-min frames (t=0.5)
python backend/infer.py --config config.yaml --t-factors 0.5

# Generate 7.5-min frames (3 frames per pair)
python backend/infer.py --config config.yaml --t-factors 0.25 0.5 0.75

# Output: ./outputs/synthetic_frames/3SIMG_AI_*_TIR1_*.h5
```

### 7. Open the dashboard
```bash
# Just open in a browser — no server needed
open frontend/index.html
```

---

## Architecture

```
I₀ (T=00:00) ──┬──► FlowEstimator ──► F₀→ₜ, F₁→ₜ ──┬──► FrameSynthesizer ──► Iₜ (T=00:15)
               │     (RAFT-lite)                        │    (U-Net + visibility masks)
I₁ (T=00:30) ──┘                                       └── warp(I₀, F₀→ₜ), warp(I₁, F₁→ₜ)
```

**Network 1 — FlowEstimator (RAFT-lite)**
- Feature pyramid (shared weights for I₀ and I₁)
- 4D correlation volume (patch matching)
- GRU update block (4 iterative refinements)
- Outputs bi-directional flow fields F₀→₁ and F₁→₀

**Network 2 — FrameSynthesizer (Super SloMo / U-Net)**
- Backward-warps I₀ and I₁ to time t using scaled flow
- Predicts per-pixel visibility masks V₀, V₁ (occlusion handling)
- U-Net refinement with skip connections
- Final blend: Iₜ = ((1-t)·V₀·I₀ᵥᵥ + t·V₁·I₁ᵥᵥ) / ((1-t)·V₀ + t·V₁)

**Loss function**
```
L = 0.84 × L1 + 0.12 × (1 - SSIM) + 0.04 × Perceptual
```

---

## Metrics

| Metric | Description              | Target  |
|--------|--------------------------|---------|
| SSIM   | Structural similarity ↑  | > 0.90  |
| PSNR   | Peak signal-to-noise ↑   | > 33 dB |
| MSE    | Mean squared error ↓     | < 0.005 |
| FSIM   | Feature similarity ↑     | > 0.89  |
| MCE    | Cloud centroid error ↓   | < 0.01  |

---

## Domain transfer: GOES-19 → INSAT-3DS

| Property        | GOES-19 ABI Ch.13  | INSAT-3DS TIR1     |
|-----------------|--------------------|--------------------|
| Wavelength      | 11.2 µm            | 10.2–11.2 µm       |
| Spatial res.    | 2 km               | 4 km               |
| Temporal res.   | 10 min             | 30 min             |
| Projection      | GOES-East FD       | Indian region      |
| Data format     | NetCDF4 (.nc)      | HDF5 (.h5)         |

The model is trained on GOES-19 (abundant 10-min ground truth) and fine-tuned / applied to INSAT-3DS.  The spectral overlap (both TIR ~11 µm) makes domain transfer viable; spatial resolution difference is handled by working in pixel space after normalisation.

---

## Citation
```
PS12 · ISRO Bharatiya Antariksh Hackathon 2026
Temporal Super-Resolution of INSAT-3DS TIR1 Imagery
Using Optical Flow Frame Interpolation
```
