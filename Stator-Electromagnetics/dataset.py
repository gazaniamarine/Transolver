"""
Dataset loading for stator electromagnetics experiment.

Each .txt file contains an unstructured FEM mesh for one stator geometry
(identified by its outer diameter). Columns: x  y  z  Ax  Ay  Az

We use only (x, y) as input and Az as the target.
z is identically zero (pure 2-D problem).
Ax, Ay are zero everywhere — so only Az carries the physics.

Train geometries : OD 28, 30, 34, 36 mm
Test  geometry   : OD 32 mm
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# File map
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Garrett_data", "angles"
)

TRAIN_IDXS = [i for i in range(31) if i != 3]
TEST_IDXS  = [3]


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------
def load_stator_file(idx: int, data_dir: str = DATA_DIR):
    """
    Read one stator .txt into numpy arrays.

    Returns
    -------
    coords    : (N, 2)  float32   — (x, y) node coordinates
    az        : (N, 1)  float32   — Az magnetic vector potential
    angle_deg : float             — Rotor rotation angle in degrees
    """
    fname = os.path.join(data_dir, f"20250826_magnetOD_32_1_Az_{idx}.txt")
    
    # Each index is shifted by 12 degrees
    angle_deg = float(idx * 12)

    # First row is the header; tab-separated
    data = np.loadtxt(fname, delimiter='\t', skiprows=1)  # (N, 6)
    coords = data[:, :2].astype(np.float32)               # x, y
    az     = data[:, 5:6].astype(np.float32)              # Az only
    return coords, az, angle_deg




# ---------------------------------------------------------------------------
# Normalisation statistics (computed on training set)
# ---------------------------------------------------------------------------
def compute_normalisation(idxs, data_dir=DATA_DIR):
    """
    Compute per-channel mean and std across all training samples
    (pooled over all nodes and all training geometries).

    Returns dict with keys: coord_mean, coord_std, az_mean, az_std
    """
    all_coords, all_az = [], []
    for i in idxs:
        coords, az, _ = load_stator_file(i, data_dir)
        all_coords.append(coords)
        all_az.append(az)

    all_coords = np.concatenate(all_coords, axis=0)
    all_az     = np.concatenate(all_az,     axis=0)

    stats = {
        "coord_mean": all_coords.mean(axis=0),    # (2,)
        "coord_std":  all_coords.std(axis=0) + 1e-8,
        "az_mean":    all_az.mean(axis=0),         # (1,)
        "az_std":     all_az.std(axis=0)  + 1e-8,
    }
    return stats


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class StatorDataset(Dataset):
    """
    One sample = one stator geometry (one .txt file).

    Returns
    -------
    coords_t : (N, 2)  normalised (x, y)
    az_t     : (N, 1)  normalised Az
    diameter : int scalar for bookkeeping
    """

    def __init__(self, idxs, stats, data_dir=DATA_DIR):
        self.samples   = []
        self.idxs      = []
        for i in idxs:
            coords, az, angle_deg = load_stator_file(i, data_dir)
            
            # Normalise variables
            coords_n = (coords - stats["coord_mean"]) / stats["coord_std"]
            az_n     = (az     - stats["az_mean"])    / stats["az_std"]
            
            # --- Encode Angular Information ---
            angle_rad = np.deg2rad(angle_deg)
            cos_val = np.cos(angle_rad)
            sin_val = np.sin(angle_rad)
            
            # Broadcast to match N nodes: shape (N, 2)
            N = coords_n.shape[0]
            angle_features = np.empty((N, 2), dtype=np.float32)
            angle_features[:, 0] = cos_val
            angle_features[:, 1] = sin_val
            
            # Concatenate coordinates and angle features: (N, 4)
            coords_out = np.concatenate([coords_n, angle_features], axis=-1)

            self.samples.append((
                torch.from_numpy(coords_out),   # (N, 4)
                torch.from_numpy(az_n),         # (N, 1)
            ))
            self.idxs.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coords, az = self.samples[idx]
        return coords, az, self.idxs[idx]


# ---------------------------------------------------------------------------
# Collate — batch samples with *different* mesh sizes
# ---------------------------------------------------------------------------
def collate_variable_mesh(batch):
    """
    Each geometry has a different N (number of mesh nodes).
    We keep them as a list of tensors so the model processes each
    geometry independently (batch_size=1 per forward pass on variable meshes).

    Returns list of (coords, az, diameter) tuples.
    """
    return batch   # list of (coords_t, az_t, diameter)


def get_dataloaders(batch_size: int = 1,
                    data_dir: str = DATA_DIR,
                    num_workers: int = 0):
    """
    Build train and test DataLoaders.

    Since mesh sizes differ per geometry, we use batch_size=1 and a custom
    collate that does not attempt to stack tensors of different shapes.
    """
    stats = compute_normalisation(TRAIN_IDXS, data_dir)

    train_ds = StatorDataset(TRAIN_IDXS, stats, data_dir)
    test_ds  = StatorDataset(TEST_IDXS,  stats, data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_variable_mesh,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_variable_mesh,
    )
    return train_loader, test_loader, stats
