"""
Training script for Transolver stator electromagnetics experiment.

Usage
-----
python train.py [--epochs 200] [--lr 1e-3] [--hidden 256] [--layers 6]
                [--heads 8] [--slice_num 32] [--mlp_ratio 2]
                [--dropout 0.0] [--gpu 0] [--seed 42]
                [--checkpoint_dir checkpoints] [--results_dir results]

Train / Test split
------------------
  Train : OD 28, 30, 34, 36 mm  (4 geometries)
  Test  : OD 32 mm              (unseen geometry)

The model receives (x, y) coordinates and predicts the Az magnetic vector
potential at every mesh node.  The mesh is unstructured and varies in size
across geometries, so we process one geometry at a time (effective batch=1).
"""

import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders, TRAIN_IDXS, TEST_IDXS
from models.Transolver import Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """‖pred − target‖₂ / ‖target‖₂ — standard neural-operator metric."""
    err  = torch.norm(pred - target)
    ref  = torch.norm(target)
    return (err / (ref + 1e-8)).item()


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_mse, total_rel = 0.0, 0.0
    n_samples = 0

    for batch in loader:
        for (coords, az, file_idx) in batch:
            coords = coords.unsqueeze(0).to(device)  # (1, N, 2)
            az     = az.unsqueeze(0).to(device)       # (1, N, 1)

            optimizer.zero_grad()
            pred = model(coords)                      # (1, N, 1)

            loss = criterion(pred, az)
            loss.backward()

            # Gradient clipping — helps with large unstructured meshes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_mse += loss.item()
            total_rel += relative_l2_error(pred.detach(), az)
            n_samples  += 1

    return total_mse / n_samples, total_rel / n_samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion, stats=None):
    model.eval()
    total_mse, total_rel = 0.0, 0.0
    n_samples = 0
    results = []

    for batch in loader:
        for (coords, az, file_idx) in batch:
            coords = coords.unsqueeze(0).to(device)   # (1, N, 2)
            az     = az.unsqueeze(0).to(device)        # (1, N, 1)

            pred  = model(coords)                      # (1, N, 1)
            mse   = criterion(pred, az).item()
            rel   = relative_l2_error(pred, az)

            total_mse += mse
            total_rel += rel
            n_samples  += 1

            # Optionally denormalise for interpretable absolute error
            if stats is not None:
                az_std  = torch.tensor(stats["az_std"],  device=device)
                az_mean = torch.tensor(stats["az_mean"], device=device)
                pred_phys = pred[0] * az_std + az_mean
                az_phys   = az[0]   * az_std + az_mean
                mae_phys  = (pred_phys - az_phys).abs().mean().item()
            else:
                mae_phys = float("nan")

            results.append({
                "file_idx":    file_idx,
                "mse":         mse,
                "rel_l2":      rel,
                "mae_physical": mae_phys,
            })

    return total_mse / n_samples, total_rel / n_samples, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Transolver stator Az-field experiment")

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument('--data_dir',       type=str,   default=None,
                        help="Override path to Garrett_data/diameter/")

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument('--hidden',         type=int,   default=256,
                        help="Hidden dimension width")
    parser.add_argument('--layers',         type=int,   default=6,
                        help="Number of Transolver blocks")
    parser.add_argument('--heads',          type=int,   default=8,
                        help="Number of attention heads (must divide hidden)")
    parser.add_argument('--slice_num',      type=int,   default=32,
                        help="Number of physics latent tokens (G)")
    parser.add_argument('--mlp_ratio',      type=int,   default=2,
                        help="FFN expansion ratio")
    parser.add_argument('--dropout',        type=float, default=0.0)

    # ── Optimiser ────────────────────────────────────────────────────────
    parser.add_argument('--epochs',         type=int,   default=500)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--warmup_epochs',  type=int,   default=10,
                        help="Linear warmup before cosine decay")

    # ── Misc ─────────────────────────────────────────────────────────────
    parser.add_argument('--gpu',            type=int,   default=0)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--val_every',      type=int,   default=10,
                        help="Run full test evaluation every N epochs")
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    parser.add_argument('--results_dir',    type=str,   default='results')

    args = parser.parse_args()
    set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────
    n_gpu     = torch.cuda.device_count()
    use_cuda  = (0 <= args.gpu < n_gpu) and torch.cuda.is_available()
    device    = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print(f"[Device] {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir,    exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    kwargs = {}
    if args.data_dir:
        kwargs['data_dir'] = args.data_dir

    train_loader, test_loader, stats = get_dataloaders(**kwargs)
    print(f"[Data] Train IDs: {TRAIN_IDXS}")
    print(f"[Data] Test  IDs: {TEST_IDXS}")
    print(f"[Data] Coord mean: {stats['coord_mean']}  std: {stats['coord_std']}")
    print(f"[Data] Az    mean: {stats['az_mean']}     std: {stats['az_std']}")

    # ── Model ────────────────────────────────────────────────────────────
    assert args.hidden % args.heads == 0, \
        f"hidden ({args.hidden}) must be divisible by heads ({args.heads})"

    model = Model(
        space_dim = 2,           # (x, y) input
        n_layers  = args.layers,
        n_hidden  = args.hidden,
        dropout   = args.dropout,
        n_head    = args.heads,
        act       = 'gelu',
        mlp_ratio = args.mlp_ratio,
        fun_dim   = 2,           # Expects cos(theta) and sin(theta) along with coords
        out_dim   = 1,           # predict scalar Az
        slice_num = args.slice_num,
    ).to(device)

    print(f"[Model] {model.__name__}")
    print(f"[Model] Trainable params: {count_params(model):,}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # Number of optimiser steps per epoch = number of training geometries
    n_train       = len(TRAIN_IDXS)
    total_steps   = args.epochs * n_train
    warmup_steps  = args.warmup_epochs * n_train

    # Cosine annealing with linear warm-up
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────────────
    history = {
        "train_mse": [], "train_rel": [],
        "test_mse":  [], "test_rel":  [],
        "test_epoch": [],
        "epoch":     [],
    }
    best_test_rel = float('inf')
    t0 = time.time()

    # Open loss log .txt — written incrementally so you can tail -f it
    loss_log_path = os.path.join(args.results_dir, "loss_log.txt")
    loss_log_fh   = open(loss_log_path, "w")
    loss_log_fh.write(
        "epoch\ttrain_mse\ttrain_rel_l2\ttest_mse\ttest_rel_l2\n")

    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs | lr={args.lr} | "
          f"hidden={args.hidden} | layers={args.layers} | "
          f"heads={args.heads} | slice_num={args.slice_num}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # -- train --
        tr_mse, tr_rel = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion)

        history["train_mse"].append(tr_mse)
        history["train_rel"].append(tr_rel)
        history["epoch"].append(epoch)

        # -- optional test evaluation --
        if epoch % args.val_every == 0 or epoch == args.epochs:
            te_mse, te_rel, te_results = evaluate(
                model, test_loader, device, criterion, stats=stats)
            history["test_mse"].append(te_mse)
            history["test_rel"].append(te_rel)
            history["test_epoch"].append(epoch)

            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train MSE={tr_mse:.6f}  RelL2={tr_rel:.4f} | "
                  f"Test  MSE={te_mse:.6f}  RelL2={te_rel:.4f} | "
                  f"MAE_phys={te_results[0]['mae_physical']:.6f} | "
                  f"t={elapsed:.1f}s")

            # Write to loss log
            loss_log_fh.write(
                f"{epoch}\t{tr_mse:.8f}\t{tr_rel:.8f}"
                f"\t{te_mse:.8f}\t{te_rel:.8f}\n")
            loss_log_fh.flush()

            # Save best checkpoint
            if te_rel < best_test_rel:
                best_test_rel = te_rel
                ckpt_path = os.path.join(
                    args.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "opt_state":   optimizer.state_dict(),
                    "test_rel_l2": te_rel,
                    "args":        vars(args),
                    "stats":       {k: v.tolist() for k, v in stats.items()},
                }, ckpt_path)
                print(f"  [Best] checkpoint saved  (rel_L2={te_rel:.6f})")
        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train MSE={tr_mse:.6f}  RelL2={tr_rel:.4f} | "
                  f"t={elapsed:.1f}s")
            # Write train-only row (no test columns)
            loss_log_fh.write(
                f"{epoch}\t{tr_mse:.8f}\t{tr_rel:.8f}\t\t\n")
            loss_log_fh.flush()

    loss_log_fh.close()

    # ── Final evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final evaluation on OD-32 (unseen test geometry)")
    print(f"{'='*60}")

    ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, final_rel, final_results = evaluate(
        model, test_loader, device, criterion, stats=stats)

    for r in final_results:
        print(f"  IDX={r['file_idx']} | "
              f"RelL2={r['rel_l2']:.6f} | "
              f"MAE(Az)={r['mae_physical']:.6e} Wb/m")

    # ── Save plain-text summary ───────────────────────────────────────────
    summary_path = os.path.join(args.results_dir, "training_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Transolver Stator Electromagnetics - Training Summary\n")
        fh.write("=" * 55 + "\n\n")
        fh.write("[Hyperparameters]\n")
        for k, v in vars(args).items():
            fh.write(f"  {k}: {v}\n")
        fh.write("\n[Final Test Results]\n")
        for r in final_results:
            fh.write(f"  IDX={r['file_idx']}\n")
            fh.write(f"    Rel-L2 error : {r['rel_l2']:.6f}\n")
            fh.write(f"    MAE (Az)     : {r['mae_physical']:.6e} Wb/m\n")
            fh.write(f"    MSE (norm.)  : {r['mse']:.6f}\n")
        fh.write("\n[Training Metrics]\n")
        fh.write(f"  Best test Rel-L2  : {best_test_rel:.6f}\n")
        fh.write(f"  Total epochs      : {args.epochs}\n")
        fh.write(f"  Loss log          : {loss_log_path}\n")

    print(f"\n[Done] Loss log    -> {loss_log_path}")
    print(f"[Done] Summary     -> {summary_path}")
    print(f"[Done] Best test Rel-L2 = {best_test_rel:.6f}")


if __name__ == "__main__":
    main()
