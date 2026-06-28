# PS12 · ISRO BAH 2026
## INSAT-3DS Temporal Super-Resolution Dashboard

> **Problem Statement 12** — Enhance the temporal resolution of INSAT-3DS TIR1 imagery from 30 min to 15 min (and optionally 7.5 min) using deep-learning-based optical flow frame interpolation.

---

## 🛰 Live Interactive Dashboard

### **[▶ https://insat-3-ds-temporal-super-resolution-lonh-7m5jx9kmz.vercel.app](https://insat-3-ds-temporal-super-resolution-lonh-7m5jx9kmz.vercel.app)**

> Fully interactive — runs in any browser, no install needed.

| Tab | What it shows |
|-----|---------------|
| 🖥 **Side-by-side viewer** | Original 30-min INSAT-3DS frames vs. AI-interpolated 15-min frames, animated with playback controls |
| 🌀 **Optical flow** | Live vector field showing cloud motion estimated by the model between frames |
| 📈 **Training curves** | Loss, SSIM, PSNR, and learning rate plotted over 100 training epochs |
| 🗂 **Source code browser** | All 7 project files with syntax highlighting, copyable, in-browser |
| ⚙ **Config panel** | Full hyperparameter, dataset, and inference configuration table |

---

## Deploy your own (Vercel — free, 1 click)

This dashboard is a **pure static HTML file** — no server, no Python, no build step.

1. Fork this repo
2. Go to [vercel.com/new](https://vercel.com/new) → Import the fork
3. Vercel auto-detects `vercel.json` → click **Deploy**
4. Done — public URL in ~30 seconds

> `vercel.json` and `.vercelignore` are pre-configured in this repo.  
> The Python backend (`backend/`, `model/`, `requirements.txt`) is for **local training only** — Vercel ignores it automatically.

---

## Project structure

```
INSAT3DS-Temporal-SuperResolution/
├── frontend/
│   └── index.html          ← Self-contained dashboard (the live site above)
├── backend/
│   ├── train.py            ← Training script (GOES-19 → model weights)
│   ├── dataset.py          ← GOES-19 (.nc) & INSAT-3DS (.h5) data loaders
│   ├── infer.py            ← Inference: generate synthetic .h5 frames
│   └── metrics.py          ← SSIM, PSNR, MSE, FSIM, motion centroid error
├── model/
│   └── model.py            ← FlowEstimator (RAFT-lite) + FrameSynthesizer (U-Net)
├── config.yaml             ← All hyperparameters and data paths
├── requirements.txt        ← Python dependencies (training only)
├── vercel.json             ← Static deployment config for Vercel
└── .vercelignore           ← Excludes Python backend from Vercel build
```

---

## Quickstart (local training)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get training data (GOES-19)
```bash
# AWS S3 public bucket — no credentials needed
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
# TensorBoard → tensorboard --logdir logs/
```

### 6. Run inference on INSAT-3DS
```bash
# 15-min frames (one synthetic frame per 30-min pair)
python backend/infer.py --config config.yaml --t-factors 0.5

# 7.5-min frames (three synthetic frames per pair)
python backend/infer.py --config config.yaml --t-factors 0.25 0.5 0.75

# Output → ./outputs/synthetic_frames/3SIMG_AI_*_TIR1_*.h5
```

### 7. Open dashboard locally
```bash
open frontend/index.html   # macOS
# or double-click it — works in any browser, no server needed
```

---

## Architecture

```
I₀ (T=00:00) ──┬──► FlowEstimator ──► F₀→ₜ, F₁→ₜ ──┬──► FrameSynthesizer ──► Iₜ (T=00:15)
               │     (RAFT-lite)                        │    (U-Net + visibility masks)
I₁ (T=00:30) ──┘                                       └── warp(I₀, F₀→ₜ), warp(I₁, F₁→ₜ)
```

**Network 1 — FlowEstimator (RAFT-lite)**
- Feature pyramid encoder (shared weights for I₀ and I₁)
- 4D correlation volume (patch matching between frames)
- GRU update block (4 iterative refinements)
- Outputs bi-directional flow fields F₀→₁ and F₁→₀

**Network 2 — FrameSynthesizer (Super SloMo / U-Net)**
- Backward-warps I₀ and I₁ to time t using scaled flow fields
- Predicts per-pixel visibility masks V₀, V₁ (occlusion handling)
- U-Net refinement with skip connections for fine detail
- Final blend: Iₜ = ((1−t)·V₀·I₀w + t·V₁·I₁w) / ((1−t)·V₀ + t·V₁)

**Loss function**
```
L = 0.84 × L1  +  0.12 × (1 − SSIM)  +  0.04 × Perceptual
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
| Coverage        | GOES-East full disk | Indian region      |
| Data format     | NetCDF4 (.nc)      | HDF5 (.h5)         |

The model trains on GOES-19 (abundant 10-min ground truth) and is applied to INSAT-3DS. The spectral overlap (~11 µm TIR in both) makes domain transfer viable; resolution differences are handled in normalised pixel space.

---

## Citation
```
PS12 · ISRO Bharatiya Antariksh Hackathon 2026
Temporal Super-Resolution of INSAT-3DS TIR1 Imagery
Using Optical Flow Frame Interpolation

GitHub    : https://github.com/Apurba-06/INSAT3DS-Temporal-SuperResolution
Dashboard : https://insat-3-ds-temporal-super-resolution-lonh-7m5jx9kmz.vercel.app
```
