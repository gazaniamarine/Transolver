# -*- coding: utf-8 -*-
"""
Quick sanity-check script - verifies data loading, model forward pass,
and gradient flow BEFORE committing to a full training run.

Run with:  python sanity_check.py [--gpu 0]
"""
import argparse
import sys
import os

import numpy as np
import torch

# Make sure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (load_stator_file, compute_normalisation,
                     get_dataloaders, TRAIN_DIAMETERS, TEST_DIAMETERS,
                     DATA_DIR)
from models.Transolver import Model


def check_data():
    print("\n" + "="*60)
    print("  [1] Data loading")
    print("="*60)
    stats = compute_normalisation(TRAIN_DIAMETERS)
    print(f"  Coord mean : {stats['coord_mean']}")
    print(f"  Coord std  : {stats['coord_std']}")
    print(f"  Az    mean : {stats['az_mean']}")
    print(f"  Az    std  : {stats['az_std']}")

    for d in TRAIN_DIAMETERS + TEST_DIAMETERS:
        c, az = load_stator_file(d)
        print(f"  OD={d}mm | nodes={len(c):6d} | "
              f"x∈[{c[:,0].min():.4f},{c[:,0].max():.4f}]  "
              f"y∈[{c[:,1].min():.4f},{c[:,1].max():.4f}]  "
              f"Az∈[{az.min():.4e},{az.max():.4e}]")
    return stats


def check_forward(device, stats):
    print("\n" + "="*60)
    print("  [2] Model forward pass")
    print("="*60)

    model = Model(
        space_dim=2, n_layers=6, n_hidden=256, n_head=8,
        slice_num=32, mlp_ratio=2, out_dim=1, dropout=0.0
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Test with mesh sizes from each geometry
    for d in TRAIN_DIAMETERS + TEST_DIAMETERS:
        c, az = load_stator_file(d)
        N = len(c)
        # Normalise
        cn  = (c  - stats["coord_mean"]) / stats["coord_std"]
        azn = (az - stats["az_mean"])    / stats["az_std"]

        x_t = torch.from_numpy(cn).unsqueeze(0).to(device)   # (1, N, 2)
        y_t = torch.from_numpy(azn).unsqueeze(0).to(device)  # (1, N, 1)

        with torch.no_grad():
            out = model(x_t)

        assert out.shape == (1, N, 1), f"Shape mismatch: {out.shape}"
        print(f"  OD={d}mm | N={N:6d} | input={tuple(x_t.shape)} "
              f"→ output={tuple(out.shape)}  ✓")


def check_backward(device, stats):
    print("\n" + "="*60)
    print("  [3] Gradient flow (1 optimiser step)")
    print("="*60)

    model = Model(
        space_dim=2, n_layers=6, n_hidden=256, n_head=8,
        slice_num=32, mlp_ratio=2, out_dim=1, dropout=0.0
    ).to(device)

    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    c, az = load_stator_file(TRAIN_DIAMETERS[0])
    cn  = (c  - stats["coord_mean"]) / stats["coord_std"]
    azn = (az - stats["az_mean"])    / stats["az_std"]

    x_t = torch.from_numpy(cn).unsqueeze(0).to(device)
    y_t = torch.from_numpy(azn).unsqueeze(0).to(device)

    opt.zero_grad()
    pred = model(x_t)
    loss = loss_fn(pred, y_t)
    loss.backward()

    # Check that gradients actually flowed
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    if no_grad:
        print(f"  [WARNING] No gradient for: {no_grad}")
    else:
        print(f"  All parameters received gradients  [OK]")

    opt.step()
    print(f"  Loss = {loss.item():.6f}  [OK]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    n_gpu    = torch.cuda.device_count()
    use_cuda = (0 <= args.gpu < n_gpu) and torch.cuda.is_available()
    device   = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print(f"Device: {device}")

    stats = check_data()
    check_forward(device, stats)
    check_backward(device, stats)

    print("\n" + "="*60)
    print("  All checks passed - ready to train!")
    print("  Run:  python train.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
