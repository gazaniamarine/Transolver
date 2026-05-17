"""
Navier-Stokes Turbulence: Transolver vs FNO
============================================
Trains Transolver on 2D turbulent Navier-Stokes data (64x64 vorticity fields)
and generates publication-quality visualizations for the TPCRL lab pitch.

Dataset:  ns_train_64.pt / ns_test_64.pt  (x: input vorticity, y: next-step vorticity)
          Located at /media/HDD/mamta_backup/datasets/fno/navier_stokes/

Author:   Adapted from Transolver (ICML 2024) codebase for turbulence demo.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PDE-Solving-StandardBenchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from utils.testloss import TestLoss
from model_dict import get_model

# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser('Transolver — 2D NS Turbulence Prediction')

parser.add_argument('--data_path', type=str,
                    default='/media/HDD/mamta_backup/datasets/fno/navier_stokes',
                    help='Path to ns_train_64.pt and ns_test_64.pt')
parser.add_argument('--resolution', type=int, default=64,
                    choices=[64, 128], help='Grid resolution (64 or 128)')

# Model
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=128)
parser.add_argument('--n-layers', type=int, default=4)
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)

# Training
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--scheduler', type=str, default='onecycle',
                    choices=['onecycle', 'step'])
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--max_grad_norm', type=float, default=1.0)

# Run config
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--eval', action='store_true', help='Evaluation only mode')
parser.add_argument('--save_name', type=str, default='ns_turbulence_transolver')
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--ntest', type=int, default=200)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

SAVE_DIR_CKPT = os.path.join('..', 'checkpoints')
SAVE_DIR_RESULTS = os.path.join('..', 'results', args.save_name)
os.makedirs(SAVE_DIR_CKPT, exist_ok=True)
os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_parameters(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n:,}")
    return n


def normalizer_stats(data):
    """Global mean / std normalization."""
    mean = data.mean()
    std  = data.std()
    return mean, std


def make_grid(H, W, device):
    """Returns (H*W, 2) position grid normalized to [0,1]."""
    gx = torch.linspace(0, 1, H, device=device)
    gy = torch.linspace(0, 1, W, device=device)
    gx, gy = torch.meshgrid(gx, gy, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (H*W, 2)


def save_vorticity_comparison(pred, gt, error, idx, epoch, save_dir, vmin=-3, vmax=3):
    """Save a 3-panel figure: Ground Truth | Prediction | Error."""
    fig = plt.figure(figsize=(15, 4))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08)

    cmap_field = 'RdBu_r'
    cmap_err   = 'hot_r'

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(gt, cmap=cmap_field, vmin=vmin, vmax=vmax, origin='lower')
    ax0.set_title('Ground Truth (DNS)', fontsize=13, fontweight='bold')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(pred, cmap=cmap_field, vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title('Transolver Prediction', fontsize=13, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[2])
    err_max = max(abs(error.min()), abs(error.max())) + 1e-8
    im2 = ax2.imshow(np.abs(error), cmap=cmap_err, vmin=0, vmax=err_max, origin='lower')
    rel_err = np.linalg.norm(error) / (np.linalg.norm(gt) + 1e-8)
    ax2.set_title(f'|Error| (Rel L2 = {rel_err:.4f})', fontsize=13, fontweight='bold')
    ax2.axis('off')

    cax = fig.add_subplot(gs[3])
    plt.colorbar(im0, cax=cax)

    fig.suptitle(f'2D NS Turbulence — Sample {idx}  |  Epoch {epoch}',
                 fontsize=14, y=1.02)
    fname = os.path.join(save_dir, f'sample_{idx:03d}_epoch_{epoch}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    return fname


def save_training_curve(train_losses, test_losses, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(train_losses, label='Train Rel-L2', linewidth=2, color='steelblue')
    ax.semilogy(test_losses,  label='Test  Rel-L2', linewidth=2, color='tomato',
                linestyle='--')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Relative L2 Loss (log scale)', fontsize=13)
    ax.set_title('Transolver Training on 2D Turbulent Navier-Stokes', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fname = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved training curve → {fname}")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    H = args.resolution
    suffix = f'_{H}'

    train_file = os.path.join(args.data_path, f'ns_train{suffix}.pt')
    test_file  = os.path.join(args.data_path, f'ns_test{suffix}.pt')

    print(f"\n[Data] Loading {H}×{H} NS data from:")
    print(f"  Train: {train_file}")
    print(f"  Test:  {test_file}")

    train_data = torch.load(train_file, map_location='cpu')
    test_data  = torch.load(test_file,  map_location='cpu')

    train_x = train_data['x'][:args.ntrain].float()   # (N, H, W)
    train_y = train_data['y'][:args.ntrain].float()
    test_x  = test_data['x'][:args.ntest].float()
    test_y  = test_data['y'][:args.ntest].float()

    # Global normalization (using training stats)
    x_mean, x_std = train_x.mean(), train_x.std()
    y_mean, y_std = train_y.mean(), train_y.std()

    train_x = (train_x - x_mean) / (x_std + 1e-8)
    train_y = (train_y - y_mean) / (y_std + 1e-8)
    test_x  = (test_x  - x_mean) / (x_std + 1e-8)
    test_y  = (test_y  - y_mean) / (y_std + 1e-8)

    # Flatten spatial dims: (N, H*W)
    train_x = train_x.reshape(args.ntrain, -1, 1)   # (N, H*W, 1)
    train_y = train_y.reshape(args.ntrain, -1, 1)
    test_x  = test_x.reshape(args.ntest,  -1, 1)
    test_y  = test_y.reshape(args.ntest,  -1, 1)

    print(f"  train_x: {train_x.shape}   train_y: {train_y.shape}")
    print(f"  test_x:  {test_x.shape}    test_y:  {test_y.shape}")

    # Positional grid
    pos = make_grid(H, H, device='cpu').unsqueeze(0)  # (1, H*W, 2)
    pos_train = pos.repeat(args.ntrain, 1, 1)
    pos_test  = pos.repeat(args.ntest, 1, 1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos_train, train_x, train_y),
        batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos_test, test_x, test_y),
        batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )

    return train_loader, test_loader, (H, y_mean.item(), y_std.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_loader, test_loader, (H, y_mean, y_std) = load_data()

    print(f"\n[Model] Building {args.model} ...")
    model = get_model(args).Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=1,              # single vorticity channel input
        out_dim=1,              # single vorticity channel output
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=bool(args.unified_pos),
        H=H, W=H
    ).cuda()

    count_parameters(model)

    loss_fn   = TestLoss(size_average=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    ckpt_path = os.path.join(SAVE_DIR_CKPT, args.save_name + '.pt')

    # ----- Evaluation only -----
    if args.eval:
        print(f"\n[Eval] Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path), strict=False)
        evaluate(model, test_loader, loss_fn, H, y_mean, y_std, epoch='final')
        return

    # ----- Training -----
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader)
    ) if args.scheduler == 'onecycle' else torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    train_history, test_history = [], []
    best_test_l2 = float('inf')

    print(f"\n[Train] Starting training for {args.epochs} epochs ...\n")
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        train_l2 = 0.0

        for pos, fx, y in train_loader:
            pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
            bsz = pos.shape[0]
            pred = model(pos, fx=fx)               # (B, H*W, 1)
            loss = loss_fn(pred.reshape(bsz, -1), y.reshape(bsz, -1))
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.scheduler == 'onecycle':
                scheduler.step()
            train_l2 += loss.item()

        if args.scheduler == 'step':
            scheduler.step()

        test_l2 = 0.0
        model.eval()
        with torch.no_grad():
            for pos, fx, y in test_loader:
                pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                bsz = pos.shape[0]
                pred = model(pos, fx=fx)
                test_l2 += loss_fn(pred.reshape(bsz, -1), y.reshape(bsz, -1)).item()

        train_l2_avg = train_l2 / args.ntrain
        test_l2_avg  = test_l2  / args.ntest
        train_history.append(train_l2_avg)
        test_history.append(test_l2_avg)

        elapsed = time.time() - t_start
        eta     = elapsed / ep * (args.epochs - ep)
        print(f"Ep {ep:4d}/{args.epochs}  "
              f"train_L2={train_l2_avg:.5f}  test_L2={test_l2_avg:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min")

        # Save best checkpoint
        if test_l2_avg < best_test_l2:
            best_test_l2 = test_l2_avg
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best test L2 = {best_test_l2:.5f} — saved checkpoint")

        # Periodic visualizations
        if ep % 50 == 0 or ep == 1:
            save_training_curve(train_history, test_history, SAVE_DIR_RESULTS)
            evaluate(model, test_loader, loss_fn, H, y_mean, y_std, epoch=ep,
                     n_samples=3, save_plots=True)

    # Final summary
    save_training_curve(train_history, test_history, SAVE_DIR_RESULTS)
    print(f"\n[Done] Best test Rel-L2 = {best_test_l2:.5f}")
    print(f"       Checkpoint saved at: {ckpt_path}")
    print(f"       Visualizations at:  {SAVE_DIR_RESULTS}/")


def evaluate(model, loader, loss_fn, H, y_mean, y_std, epoch,
             n_samples=5, save_plots=True):
    """Run full test evaluation and optionally save vorticity comparison plots."""
    model.eval()
    total_l2  = 0.0
    n_total   = 0
    shown     = 0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for pos, fx, y in loader:
            pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
            bsz = pos.shape[0]
            pred = model(pos, fx=fx)                           # (B, H*W, 1)
            total_l2 += loss_fn(pred.reshape(bsz, -1),
                                y.reshape(bsz, -1)).item()

            # Denormalize for visualization
            pred_np = (pred[:, :, 0].cpu().numpy() * y_std + y_mean)
            gt_np   = (y[:, :, 0].cpu().numpy()   * y_std + y_mean)

            all_preds.append(pred_np)
            all_gts.append(gt_np)
            n_total += bsz

            if save_plots and shown < n_samples:
                for i in range(min(n_samples - shown, bsz)):
                    p = pred_np[i].reshape(H, H)
                    g = gt_np[i].reshape(H, H)
                    e = p - g
                    save_vorticity_comparison(
                        p, g, e,
                        idx=shown + i,
                        epoch=epoch,
                        save_dir=SAVE_DIR_RESULTS
                    )
                shown += bsz

    avg_l2 = total_l2 / n_total
    print(f"  [Eval epoch={epoch}] Test Rel-L2 = {avg_l2:.5f}  (N={n_total})")

    # Save statistical summary
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts   = np.concatenate(all_gts, axis=0)
    np.save(os.path.join(SAVE_DIR_RESULTS, 'predictions.npy'), all_preds)
    np.save(os.path.join(SAVE_DIR_RESULTS, 'ground_truth.npy'), all_gts)

    return avg_l2


if __name__ == '__main__':
    main()
