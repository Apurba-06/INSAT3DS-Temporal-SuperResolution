# metrics.py — PS12 · ISRO BAH 2026
# Validation metrics: SSIM (differentiable + numpy), PSNR, MSE, FSIM
# Used both during training (gradient-compatible) and evaluation (numpy).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ── Differentiable SSIM loss (used during training) ─────────────
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss — differentiable via PyTorch.
    Returns 1 - SSIM so it can be minimised as a loss.

    Formula:
        SSIM(x, y) = (2μₓμᵧ + C1)(2σₓᵧ + C2)
                     ─────────────────────────────
                     (μₓ² + μᵧ² + C1)(σₓ² + σᵧ² + C2)
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.pad         = window_size // 2
        self.C1          = (0.01) ** 2
        self.C2          = (0.03) ** 2
        # Build Gaussian kernel
        g = torch.arange(window_size).float() - window_size // 2
        gauss  = torch.exp(-(g ** 2) / (2 * sigma ** 2))
        gauss /= gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)           # (W, W)
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
        self.register_buffer('kernel', kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, C, H, W) in [0, 1]
        Returns scalar loss (1 - mean SSIM).
        """
        K = self.kernel.to(x.device)
        mu_x  = F.conv2d(x, K, padding=self.pad, groups=self.channels)
        mu_y  = F.conv2d(y, K, padding=self.pad, groups=self.channels)
        mu_xx = mu_x * mu_x
        mu_yy = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sig_x  = F.conv2d(x * x, K, padding=self.pad, groups=self.channels) - mu_xx
        sig_y  = F.conv2d(y * y, K, padding=self.pad, groups=self.channels) - mu_yy
        sig_xy = F.conv2d(x * y, K, padding=self.pad, groups=self.channels) - mu_xy
        num = (2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)
        den = (mu_xx + mu_yy + self.C1) * (sig_x + sig_y + self.C2)
        ssim_map = num / (den + 1e-8)
        return 1.0 - ssim_map.mean()


# ── Numpy metrics (used during evaluation) ───────────────────────
def ssim_np(pred: np.ndarray, gt: np.ndarray,
            window: int = 11, sigma: float = 1.5) -> float:
    """Compute SSIM between two single-channel float32 arrays in [0,1]."""
    from scipy.ndimage import gaussian_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p  = gaussian_filter(pred, sigma)
    mu_g  = gaussian_filter(gt,   sigma)
    mu_pp = gaussian_filter(pred * pred, sigma) - mu_p ** 2
    mu_gg = gaussian_filter(gt   * gt,   sigma) - mu_g ** 2
    mu_pg = gaussian_filter(pred * gt,   sigma) - mu_p * mu_g
    num = (2 * mu_p * mu_g + C1) * (2 * mu_pg + C2)
    den = (mu_p ** 2 + mu_g ** 2 + C1) * (mu_pp + mu_gg + C2)
    return float(np.mean(num / (den + 1e-8)))


def psnr_np(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def mse_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((pred - gt) ** 2))


def fsim_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Feature Similarity Index (FSIM) — simplified implementation.
    Uses Phase Congruency as the primary feature map.
    Full implementation would use the phasepack library;
    this version uses gradient magnitude as a proxy.
    """
    def gradient_mag(img):
        gy = np.gradient(img, axis=0)
        gx = np.gradient(img, axis=1)
        return np.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    PC_p = gradient_mag(pred)
    PC_g = gradient_mag(gt)
    T1, T2 = 0.85, 160.0 / 255.0
    S_PC = (2 * PC_p * PC_g + T1) / (PC_p ** 2 + PC_g ** 2 + T1)
    S_GM = (2 * PC_p * PC_g + T2) / (PC_p ** 2 + PC_g ** 2 + T2)
    PC_m = np.maximum(PC_p, PC_g)
    fsim = np.sum(S_PC * S_GM * PC_m) / (np.sum(PC_m) + 1e-8)
    return float(fsim)


def motion_centroid_error(pred: np.ndarray, gt: np.ndarray,
                          threshold: float = 0.6) -> float:
    """
    Domain-specific metric: distance between cloud cluster centroids.
    Thresholds both images to isolate bright (cold, high) cloud regions,
    then computes distance between their centroids of mass.

    Lower is better. Units: normalised image coordinates [0, 1].
    """
    def centroid(img, thr):
        mask = img > thr
        if mask.sum() == 0:
            return np.array([0.5, 0.5])
        ys, xs = np.where(mask)
        H, W = img.shape
        return np.array([ys.mean() / H, xs.mean() / W])
    cp = centroid(pred, threshold)
    cg = centroid(gt,   threshold)
    return float(np.linalg.norm(cp - cg))


# ── Batch evaluation (called from train.py) ──────────────────────
@torch.no_grad()
def compute_metrics(model: torch.nn.Module,
                    loader,
                    device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a DataLoader.
    Returns dict with keys: ssim, psnr, mse, fsim
    """
    model.eval()
    ssim_vals, psnr_vals, mse_vals, fsim_vals = [], [], [], []

    for batch in loader:
        I0 = batch['frame0'].to(device)
        I1 = batch['frame1'].to(device)
        It = batch['frame_t'].to(device)
        t  = batch['t'].to(device)

        pred = model(I0, I1, t).clamp(0, 1)

        # Convert to numpy for metric computation (per sample in batch)
        pred_np = pred.cpu().numpy()[:, 0]   # (B, H, W)
        gt_np   = It.cpu().numpy()[:, 0]

        for p, g in zip(pred_np, gt_np):
            ssim_vals.append(ssim_np(p, g))
            psnr_vals.append(psnr_np(p, g))
            mse_vals.append(mse_np(p, g))
            fsim_vals.append(fsim_np(p, g))

    return {
        'ssim': float(np.mean(ssim_vals)),
        'psnr': float(np.mean(psnr_vals)),
        'mse':  float(np.mean(mse_vals)),
        'fsim': float(np.mean(fsim_vals)),
    }


@torch.no_grad()
def compute_metrics_pair(pred: torch.Tensor,
                         gt: torch.Tensor) -> Dict[str, float]:
    """Compute metrics for a single predicted/GT pair (tensors)."""
    p = pred.clamp(0, 1).cpu().numpy()[0, 0]
    g = gt.cpu().numpy()[0, 0]
    return {
        'ssim': ssim_np(p, g),
        'psnr': psnr_np(p, g),
        'mse':  mse_np(p, g),
        'fsim': fsim_np(p, g),
        'mce':  motion_centroid_error(p, g),
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Metrics self-test:")
    a = np.random.rand(256, 256).astype(np.float32)
    b = a + np.random.randn(256, 256).astype(np.float32) * 0.05

    print(f"  SSIM : {ssim_np(a, b):.4f}  (identical → 1.0)")
    print(f"  PSNR : {psnr_np(a, b):.2f} dB")
    print(f"  MSE  : {mse_np(a, b):.6f}")
    print(f"  FSIM : {fsim_np(a, b):.4f}")
    print(f"  MCE  : {motion_centroid_error(a, b):.6f}")
    print(f"  SSIM (identical): {ssim_np(a, a):.4f}")

    # Differentiable loss test
    loss_fn = SSIMLoss()
    x = torch.rand(2, 1, 64, 64)
    y = x + torch.randn_like(x) * 0.05
    loss = loss_fn(x, y)
    print(f"  SSIMLoss (torch): {loss.item():.4f}")
