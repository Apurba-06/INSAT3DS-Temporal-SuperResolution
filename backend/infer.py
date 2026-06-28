# infer.py — PS12 · ISRO BAH 2026
# Inference script: load trained model, run on INSAT-3DS frame pairs,
# save synthetic intermediate frames in INSAT-3DS .h5 format.

import yaml, logging, argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TemporalInterpolationNet
from dataset import INSAT3DSDataset, denormalise_bt
from metrics import compute_metrics_pair

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ── Core interpolation function ──────────────────────────────────
@torch.no_grad()
def interpolate_pair(model: torch.nn.Module,
                     I0: torch.Tensor,
                     I1: torch.Tensor,
                     t: float = 0.5,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Generate one synthetic frame between I0 and I1.

    Args:
        I0:  (1, H, W) tensor, normalised [0,1]
        I1:  (1, H, W) tensor, normalised [0,1]
        t:   interpolation factor 0 < t < 1
             t=0.5 → midpoint (15 min between 30-min frames)
             t=0.25 → 7.5 min, t=0.75 → 22.5 min

    Returns:
        It:  (1, H, W) synthesized frame, normalised [0,1]
    """
    model.eval()
    I0t = I0.unsqueeze(0).to(device)   # add batch dim
    I1t = I1.unsqueeze(0).to(device)
    tv  = torch.tensor([[t]], dtype=torch.float32, device=device)
    It  = model(I0t, I1t, tv)
    return It.squeeze(0).cpu().clamp(0.0, 1.0)


# ── Output .h5 writer ────────────────────────────────────────────
def save_synthetic_frame(frame_norm: np.ndarray,
                         out_path: str,
                         t_factor: float,
                         source_path0: str,
                         source_path1: str,
                         bt_min: float = 180.0,
                         bt_max: float = 320.0) -> None:
    """
    Save synthesized frame as INSAT-3DS TIR1 compatible .h5 file.

    Stores:
      - IMG_TIR1:  brightness temperature array (K), float32
      - Metadata attributes describing generation provenance
    """
    bt = denormalise_bt(frame_norm, bt_min, bt_max)
    with h5py.File(out_path, 'w') as f:
        ds = f.create_dataset('IMG_TIR1', data=bt,
                              compression='gzip', compression_opts=4)
        # CF-convention attributes
        ds.attrs['long_name']    = 'AI-Synthesized TIR1 Brightness Temperature'
        ds.attrs['units']        = 'K'
        ds.attrs['valid_range']  = np.array([bt_min, bt_max], dtype=np.float32)
        ds.attrs['_FillValue']   = np.float32(-999.0)
        # Provenance
        f.attrs['Conventions']        = 'CF-1.8'
        f.attrs['source']             = 'PS12-ISRO-BAH2026 Temporal Super-Resolution'
        f.attrs['model']              = 'TemporalInterpolationNet (RAFT+UNet)'
        f.attrs['interpolation_t']    = float(t_factor)
        f.attrs['frame0_source']      = str(source_path0)
        f.attrs['frame1_source']      = str(source_path1)
        f.attrs['generated_at']       = datetime.utcnow().isoformat() + 'Z'
        f.attrs['note']               = (
            f'Synthetic frame at t={t_factor:.3f} between two real observations. '
            'Not a real satellite observation.'
        )


# ── Parse timestamp from INSAT filename ─────────────────────────
def parse_insat_time(path: str) -> Optional[datetime]:
    """
    Parse UTC time from MOSDAC filename:
      3SIMG_20240601_0030_L1B_STD_V01R00_TIR1.h5
               ^^^^^^^^ ^^^^
    """
    try:
        stem = Path(path).stem
        parts = stem.split('_')
        date_str = parts[1]       # e.g. 20240601
        time_str = parts[2]       # e.g. 0030
        return datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
    except Exception:
        return None


def synthetic_filename(t0: datetime, t1: datetime, t_factor: float) -> str:
    """Generate output filename for a synthetic frame."""
    dt = (t1 - t0) * t_factor
    ts = t0 + dt
    return f"3SIMG_AI_{ts.strftime('%Y%m%d_%H%M')}_TIR1_t{t_factor:.2f}.h5"


# ── Main inference loop ──────────────────────────────────────────
def run_inference(cfg_path: str = 'config.yaml',
                  t_factors: list = None) -> None:
    """
    Run full inference on INSAT-3DS data directory.

    t_factors: list of interpolation values to generate.
               Default [0.5] → one 15-min frame per pair.
               Use [0.25, 0.5, 0.75] for 7.5-min effective cadence.
    """
    if t_factors is None:
        t_factors = [0.5]

    cfg    = yaml.safe_load(open(cfg_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device} | t_factors: {t_factors}")

    # Load model
    model   = TemporalInterpolationNet(cfg).to(device)
    ckpt    = Path(cfg['output']['checkpoint_dir']) / 'best_model.pth'
    state   = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    log.info(f"Loaded checkpoint: {ckpt}")

    # Dataset & output
    ds      = INSAT3DSDataset(cfg['data']['insat_root'])
    loader  = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    out_dir = Path(cfg['output']['dir']); out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Processing {len(ds)} frame pairs → {out_dir}")

    all_metrics = {t: [] for t in t_factors}

    for batch in tqdm(loader, desc='Interpolating INSAT-3DS'):
        I0       = batch['frame0'][0]    # (1, H, W)
        I1       = batch['frame1'][0]
        path0    = batch['path0'][0]
        path1    = batch['path1'][0]

        t0 = parse_insat_time(path0)
        t1 = parse_insat_time(path1)

        for t in t_factors:
            It_tensor = interpolate_pair(model, I0, I1, t=t, device=device)
            It_np     = It_tensor.numpy()[0]   # (H, W)

            fname    = synthetic_filename(t0, t1, t) if (t0 and t1) else f"synthetic_t{t:.2f}_{hash(path0)}.h5"
            out_path = out_dir / fname
            save_synthetic_frame(It_np, str(out_path), t, path0, path1)
            all_metrics[t].append({'path': str(out_path)})

    # Summary
    for t, results in all_metrics.items():
        log.info(f"t={t:.2f} → {len(results)} synthetic frames saved")
    log.info(f"All outputs in: {out_dir}")


# ── Entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PS12 INSAT-3DS Temporal Interpolation Inference')
    parser.add_argument('--config',    default='config.yaml',  help='Path to config.yaml')
    parser.add_argument('--t-factors', nargs='+', type=float,  default=[0.5],
                        help='Interpolation factors (e.g. 0.25 0.5 0.75 for 7.5-min cadence)')
    args = parser.parse_args()
    run_inference(cfg_path=args.config, t_factors=args.t_factors)
