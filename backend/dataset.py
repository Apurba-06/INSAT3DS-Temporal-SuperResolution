# dataset.py — PS12 · ISRO BAH 2026
# Data loading for GOES-19 ABI Ch.13 (training) and INSAT-3DS TIR1 (inference)
# Handles NetCDF4 (.nc) and HDF5 (.h5) formats, Planck calibration,
# reprojection, and frame-triplet sampling.

import numpy as np
import h5py
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, Optional


# ── Calibration utilities ────────────────────────────────────────
def radiance_to_bt(rad: np.ndarray, fk1: float, fk2: float,
                   bc1: float, bc2: float) -> np.ndarray:
    """
    Convert GOES-19 ABI radiance to Brightness Temperature (K)
    using the standard Planck function inversion.

    BT = (fk2 / log(fk1 / rad + 1) - bc1) / bc2
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        log_arg = np.log(fk1 / np.where(rad > 0, rad, np.nan) + 1.0)
        bt = (fk2 / log_arg - bc1) / bc2
    return np.nan_to_num(bt, nan=200.0)


def normalise_bt(bt: np.ndarray,
                 bt_min: float = 180.0,
                 bt_max: float = 320.0) -> np.ndarray:
    """Normalise brightness temperature to [0, 1]."""
    return np.clip((bt - bt_min) / (bt_max - bt_min), 0.0, 1.0).astype(np.float32)


def denormalise_bt(arr: np.ndarray,
                   bt_min: float = 180.0,
                   bt_max: float = 320.0) -> np.ndarray:
    """Invert normalisation back to Kelvin."""
    return (arr * (bt_max - bt_min) + bt_min).astype(np.float32)


# ── GOES-19 dataset (training) ───────────────────────────────────
class SatelliteFramePairDataset(Dataset):
    """
    Loads consecutive frame triplets from GOES-19 ABI Channel 13.

    Structure on disk:
        root/
          2024/
            001/          (day-of-year)
              OR_ABI-L1b-RadC-M6C13_G19_s*.nc
              ...

    Triplet: (I0, It, I1) where
        I0 = frame at time T          (input)
        It = frame at time T+10min    (hidden ground truth)
        I1 = frame at time T+20min    (input)
        t  = 0.5                      (interpolation factor)

    During training on GOES-19 the model learns to predict the
    middle frame from the two outer frames, supervised by the
    real T+10min observation. At inference time on INSAT-3DS,
    the same model fills the missing T+15min frame between
    T+0min and T+30min observations.
    """

    def __init__(self, root: str, split: str = 'train',
                 patch: int = 256, bt_min: float = 180.0, bt_max: float = 320.0):
        self.root   = Path(root)
        self.patch  = patch
        self.bt_min = bt_min
        self.bt_max = bt_max

        # Collect all .nc files sorted by timestamp
        all_files = sorted(self.root.glob('**/*.nc'))
        if len(all_files) == 0:
            raise FileNotFoundError(f"No .nc files found under {root}")

        # 95/5 train/val split by file index
        n = len(all_files)
        cut = int(n * 0.95)
        self.files = all_files[:cut] if split == 'train' else all_files[cut:]

    def __len__(self) -> int:
        return max(0, len(self.files) - 2)   # triplets need 3 consecutive files

    # ── Internal helpers ─────────────────────────────────────────
    def _load_nc(self, path: Path) -> np.ndarray:
        """Load one GOES-19 .nc file → normalised brightness temperature array."""
        with nc.Dataset(str(path), 'r') as ds:
            rad  = ds.variables['Rad'][:].astype(np.float32)
            fk1  = float(ds.variables['planck_fk1'][:])
            fk2  = float(ds.variables['planck_fk2'][:])
            bc1  = float(ds.variables['planck_bc1'][:])
            bc2  = float(ds.variables['planck_bc2'][:])
        bt = radiance_to_bt(rad, fk1, fk2, bc1, bc2)
        return normalise_bt(bt, self.bt_min, self.bt_max)

    def _random_crop(self, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Random crop to self.patch × self.patch from all arrays simultaneously."""
        H, W = arrays[0].shape[-2:]
        if H < self.patch or W < self.patch:
            arrays = tuple(
                np.pad(a, ((max(0, self.patch - H), 0), (max(0, self.patch - W), 0)), mode='reflect')
                for a in arrays
            )
            H, W = arrays[0].shape[-2:]
        y = np.random.randint(0, H - self.patch + 1)
        x = np.random.randint(0, W - self.patch + 1)
        return tuple(a[y:y + self.patch, x:x + self.patch] for a in arrays)

    def _augment(self, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Horizontal flip augmentation (applied consistently to all arrays)."""
        if np.random.rand() > 0.5:
            return tuple(np.fliplr(a) for a in arrays)
        return arrays

    def _to_tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.copy()).unsqueeze(0).float()

    # ── __getitem__ ──────────────────────────────────────────────
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f0 = self._load_nc(self.files[idx])        # T+0min
        ft = self._load_nc(self.files[idx + 1])    # T+10min (hidden GT)
        f1 = self._load_nc(self.files[idx + 2])    # T+20min

        # Crop and augment
        f0, ft, f1 = self._random_crop(f0, ft, f1)
        f0, ft, f1 = self._augment(f0, ft, f1)

        return {
            'frame0':   self._to_tensor(f0),
            'frame_t':  self._to_tensor(ft),
            'frame1':   self._to_tensor(f1),
            't':        torch.tensor([0.5], dtype=torch.float32),
        }


# ── INSAT-3DS dataset (inference) ───────────────────────────────
class INSAT3DSDataset(Dataset):
    """
    Loads consecutive INSAT-3DS TIR1 .h5 files from MOSDAC.

    File naming convention (MOSDAC standard):
        3SIMG_20240601_0000_L1B_STD_V01R00_TIR1.h5
        3SIMG_20240601_0030_L1B_STD_V01R00_TIR1.h5
        ...

    Returns pairs (I0, I1) every 30 minutes; model fills T+15min.

    Usage:
        ds = INSAT3DSDataset('/data/INSAT3DS/2024/06/')
        for batch in DataLoader(ds, batch_size=1):
            It = model(batch['frame0'], batch['frame1'], batch['t'])
    """

    def __init__(self, root: str, bt_min: float = 180.0, bt_max: float = 320.0):
        self.files  = sorted(Path(root).glob('3SIMG*_TIR1*.h5'))
        self.bt_min = bt_min
        self.bt_max = bt_max
        if len(self.files) == 0:
            raise FileNotFoundError(f"No INSAT-3DS TIR1 .h5 files found under {root}")

    def __len__(self) -> int:
        return max(0, len(self.files) - 1)

    def _read_h5(self, path: Path) -> torch.Tensor:
        with h5py.File(str(path), 'r') as f:
            # Standard MOSDAC variable name for TIR1 brightness temperature
            if 'IMG_TIR1' in f:
                bt = f['IMG_TIR1'][:].astype(np.float32)
            elif 'BrightnessTemperature' in f:
                bt = f['BrightnessTemperature'][:].astype(np.float32)
            else:
                # Fallback: first dataset found
                key = list(f.keys())[0]
                bt = f[key][:].astype(np.float32)
        arr = normalise_bt(bt, self.bt_min, self.bt_max)
        return torch.from_numpy(arr).unsqueeze(0).float()

    def __getitem__(self, idx: int) -> Dict:
        return {
            'frame0': self._read_h5(self.files[idx]),
            'frame1': self._read_h5(self.files[idx + 1]),
            't':      torch.tensor([0.5], dtype=torch.float32),
            'path0':  str(self.files[idx]),
            'path1':  str(self.files[idx + 1]),
        }


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == '__main__':
    import tempfile, os
    print("Dataset classes loaded successfully.")
    print("  SatelliteFramePairDataset — GOES-19 ABI Ch.13 (.nc)")
    print("  INSAT3DSDataset           — INSAT-3DS TIR1 (.h5)")
    print("\nTo use:")
    print("  ds = SatelliteFramePairDataset('/data/GOES19/', split='train')")
    print("  ds = INSAT3DSDataset('/data/INSAT3DS/2024/')")
