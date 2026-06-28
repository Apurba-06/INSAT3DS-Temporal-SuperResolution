# train.py — PS12 · ISRO BAH 2026
# Main training script for INSAT-3DS temporal super-resolution
# Trains on GOES-19 ABI Channel 13 (10-min ground truth),
# then fine-tunes on INSAT-3DS TIR1 characteristics.

import os, yaml, argparse, logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import TemporalInterpolationNet
from dataset import SatelliteFramePairDataset
from metrics import SSIMLoss, compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ── Loss function ────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    L_total = w_l1 * L1 + w_ssim * (1 - SSIM) + w_perc * Perceptual
    Weights from config: 0.84 / 0.12 / 0.04
    """
    def __init__(self, cfg):
        super().__init__()
        self.w_l1   = cfg['train']['loss_l1']
        self.w_ssim = cfg['train']['loss_ssim']
        self.ssim   = SSIMLoss()
        self.l1     = nn.L1Loss()

    def forward(self, pred, target):
        l1   = self.l1(pred, target)
        ssim = self.ssim(pred, target)
        return self.w_l1 * l1 + self.w_ssim * ssim, {'l1': l1.item(), 'ssim': ssim.item()}


# ── Training loop ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        I0 = batch['frame0'].to(device)    # T = 00:00
        It = batch['frame_t'].to(device)   # T = 00:10 (hidden GT)
        I1 = batch['frame1'].to(device)    # T = 00:20
        t  = batch['t'].to(device)         # interpolation factor (0.5)

        pred = model(I0, I1, t)
        loss, comps = criterion(pred, It)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += loss.item(); n += 1
        if i % 50 == 0:
            log.info(f"  Epoch {epoch:03d} step {i:04d} | loss={loss.item():.4f} "
                     f"l1={comps['l1']:.4f} ssim={comps['ssim']:.4f}")

    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        I0 = batch['frame0'].to(device)
        It = batch['frame_t'].to(device)
        I1 = batch['frame1'].to(device)
        t  = batch['t'].to(device)
        pred = model(I0, I1, t)
        loss, _ = criterion(pred, It)
        total += loss.item(); n += 1
    return total / max(n, 1)


# ── Main ─────────────────────────────────────────────────────────
def main(cfg_path: str = 'config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg['train'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Directories
    ckpt_dir = Path(cfg['output']['checkpoint_dir']); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(cfg['output']['log_dir']);        log_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir)

    # Datasets
    train_ds = SatelliteFramePairDataset(cfg['data']['train_root'], split='train',
                                         patch=cfg['train']['patch_size'])
    val_ds   = SatelliteFramePairDataset(cfg['data']['train_root'], split='val',
                                         patch=cfg['train']['patch_size'])
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=4,
                              shuffle=False, num_workers=2, pin_memory=True)
    log.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model, optimizer, scheduler
    model     = TemporalInterpolationNet(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'],
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['train']['epochs'], eta_min=1e-6)
    criterion = CombinedLoss(cfg)

    # Resume from checkpoint if available
    best_ssim, start_epoch = 0.0, 0
    resume = ckpt_dir / 'latest.pth'
    if resume.exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_ssim   = ckpt.get('best_ssim', 0.0)
        start_epoch = ckpt.get('epoch', 0)
        log.info(f"Resumed from epoch {start_epoch}, best SSIM={best_ssim:.4f}")

    # Training loop
    for epoch in range(start_epoch, cfg['train']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)
        val_loss   = validate(model, val_loader, criterion, device)
        metrics    = compute_metrics(model, val_loader, device)
        scheduler.step()

        log.info(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                 f"SSIM={metrics['ssim']:.4f} PSNR={metrics['psnr']:.2f} dB MSE={metrics['mse']:.5f}")

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Metrics/SSIM', metrics['ssim'], epoch)
        writer.add_scalar('Metrics/PSNR', metrics['psnr'], epoch)
        writer.add_scalar('Metrics/MSE',  metrics['mse'],  epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Save latest checkpoint
        torch.save({'epoch': epoch + 1, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_ssim': best_ssim},
                   ckpt_dir / 'latest.pth')

        # Save best model
        if metrics['ssim'] > best_ssim:
            best_ssim = metrics['ssim']
            torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
            log.info(f"  ✓ New best SSIM={best_ssim:.4f} — saved best_model.pth")

    writer.close()
    log.info(f"Training complete. Best SSIM={best_ssim:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
