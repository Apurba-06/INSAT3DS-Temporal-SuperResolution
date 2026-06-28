# model.py — PS12 · ISRO BAH 2026
# Two-network temporal interpolation architecture:
#   Network 1: RAFT-lite optical flow estimator (bi-directional)
#   Network 2: U-Net frame synthesizer with visibility masks

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FeaturePyramid(nn.Module):
    """Multi-scale feature extraction for both input frames."""
    def __init__(self, hidden=64):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock(1, hidden, k=7),
            ConvBlock(hidden, hidden * 2, s=2),
            ConvBlock(hidden * 2, hidden * 4, s=2),
            ConvBlock(hidden * 4, hidden * 4),
        ])

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats  # multi-scale features


class FlowEstimator(nn.Module):
    """
    RAFT-lite: correlation volume + GRU update block.
    Outputs bi-directional flow fields F01 (I0→I1) and F10 (I1→I0).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.feat_net = FeaturePyramid(hidden)
        # Correlation head: takes concatenated features from both frames
        self.corr_head = nn.Sequential(
            ConvBlock(hidden * 8, hidden * 4),
            ConvBlock(hidden * 4, hidden * 2),
        )
        # GRU update block (simplified)
        self.gru = nn.GRUCell(hidden * 2 * 4, hidden * 2)
        # Flow projection: outputs 4 channels (F01_x, F01_y, F10_x, F10_y)
        self.flow_head = nn.Conv2d(hidden * 2, 4, 1)

    def forward(self, I0, I1):
        f0 = self.feat_net(I0)[-1]   # take deepest feature
        f1 = self.feat_net(I1)[-1]
        corr = torch.cat([f0, f1], dim=1)
        feat = self.corr_head(corr)
        flow = self.flow_head(feat)
        # Upsample back to original resolution
        flow = F.interpolate(flow, size=I0.shape[-2:], mode='bilinear', align_corners=False)
        F01, F10 = flow[:, :2], flow[:, 2:]
        return F01, F10


class FrameSynthesizer(nn.Module):
    """
    U-Net that takes warped frames + flow fields as input,
    produces refined synthetic frame Iₜ + visibility masks V0, V1.
    Visibility masks handle occlusion: per-pixel trust weights
    controlling how much each warped frame contributes at each location.
    """
    def __init__(self, hidden=64):
        super().__init__()
        # Input: I0_warped(1) + I1_warped(1) + F01t(2) + F10t(2) = 6 channels
        self.enc1 = ConvBlock(6, hidden)
        self.enc2 = ConvBlock(hidden, hidden * 2, s=2)
        self.enc3 = ConvBlock(hidden * 2, hidden * 4, s=2)
        self.bottleneck = ConvBlock(hidden * 4, hidden * 4)
        # Decoder with skip connections
        self.dec3 = ConvBlock(hidden * 8, hidden * 2)
        self.dec2 = ConvBlock(hidden * 4, hidden)
        self.dec1 = ConvBlock(hidden * 2, hidden)
        # Output: Iₜ_residual(1) + V0(1) + V1(1)
        self.out = nn.Conv2d(hidden, 3, 1)

    def forward(self, I0w, I1w, F01t, F10t):
        x = torch.cat([I0w, I1w, F01t, F10t], dim=1)  # (B, 6, H, W)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Bottleneck
        b = self.bottleneck(e3)
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([b, e3], dim=1))
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        out = self.out(d1)
        It_res = out[:, :1]
        V0 = torch.sigmoid(out[:, 1:2])   # visibility mask for I0_warped
        V1 = torch.sigmoid(out[:, 2:])    # visibility mask for I1_warped
        return It_res, V0, V1


def backward_warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward warp: sample frame at locations specified by flow.
    frame: (B, C, H, W)
    flow:  (B, 2, H, W) — pixel displacements (dx, dy)
    """
    B, C, H, W = frame.shape
    # Build sampling grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=frame.device).float(),
        torch.arange(W, device=frame.device).float(),
        indexing='ij'
    )
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid + flow
    # Normalize to [-1, 1]
    grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
    return F.grid_sample(
        frame, grid.permute(0, 2, 3, 1),
        align_corners=True, padding_mode='border', mode='bilinear'
    )


class TemporalInterpolationNet(nn.Module):
    """
    Full end-to-end temporal interpolation network.

    Given:
        I0: satellite frame at time T
        I1: satellite frame at time T + Δt
        t:  interpolation factor ∈ (0, 1)

    Produces:
        Iₜ: synthesized frame at time T + t·Δt

    For INSAT-3DS:  Δt = 30 min, t = 0.5  →  fills T + 15 min
    For 7.5-min output: run twice with t = 0.25 and t = 0.75
    """
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg['model']['hidden_channels']
        self.flow_net  = FlowEstimator(h)
        self.synth_net = FrameSynthesizer(h)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1) interpolation factor
        t_val = t.view(-1, 1, 1, 1)

        # ── Step 1: Estimate bi-directional optical flow ────────
        F01, F10 = self.flow_net(I0, I1)
        # F01: flow from I0 to I1 (I0's pixels move in this direction)
        # F10: flow from I1 to I0

        # ── Step 2: Scale flows to interpolation time t ─────────
        # At t=0, no warp from I0; at t=1, full warp from I0
        F01t = -t_val * F01          # I0 → Iₜ  (fraction of full motion)
        F10t = -(1 - t_val) * F10    # I1 → Iₜ

        # ── Step 3: Warp both frames to time t ──────────────────
        I0_warped = backward_warp(I0, F01t)
        I1_warped = backward_warp(I1, F10t)

        # ── Step 4: Synthesize + visibility masks ────────────────
        It_res, V0, V1 = self.synth_net(I0_warped, I1_warped, F01t, F10t)

        # ── Step 5: Blend with occlusion-aware weighted average ──
        # Numerator: time-weighted blend of warped frames, gated by visibility
        numerator   = (1 - t_val) * V0 * I0_warped + t_val * V1 * I1_warped
        denominator = (1 - t_val) * V0 + t_val * V1 + 1e-6

        It_blend = numerator / denominator

        # ── Step 6: Add residual from synthesizer (fine detail) ──
        return It_blend + It_res * 0.15


# ── Quick sanity check ──────────────────────────────────────────
if __name__ == '__main__':
    cfg = {'model': {'hidden_channels': 64}}
    model = TemporalInterpolationNet(cfg)
    I0 = torch.randn(2, 1, 256, 256)
    I1 = torch.randn(2, 1, 256, 256)
    t  = torch.tensor([[0.5], [0.5]])
    out = model(I0, I1, t)
    print(f"Input:  {I0.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
