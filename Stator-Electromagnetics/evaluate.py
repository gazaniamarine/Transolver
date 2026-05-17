"""
Inference & visualisation for the trained stator Transolver.

Usage
-----
python evaluate.py [--checkpoint checkpoints/best_model.pth]
                   [--gpu 0]
                   [--results_dir results]
                   [--loss_log  results/loss_log.txt]
                   [--save_fields]

Produces
--------
  results/od32_prediction.png    -- 3-panel scatter plot (truth | pred | error)
  results/od32_fields.txt        -- tab-separated: x  y  Az_true  Az_pred  abs_error
  results/loss_curves.png        -- 2-panel MSE + Rel-L2 vs epoch
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import (StatorDataset, compute_normalisation,
                     TRAIN_IDXS, TEST_IDXS, DATA_DIR)
from models.Transolver import Model


# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt  = torch.load(ckpt_path, map_location=device)
    args  = ckpt["args"]
    stats = {k: np.array(v) for k, v in ckpt["stats"].items()}

    model = Model(
        space_dim = 2,
        n_layers  = args["layers"],
        n_hidden  = args["hidden"],
        dropout   = args["dropout"],
        n_head    = args["heads"],
        act       = "gelu",
        mlp_ratio = args["mlp_ratio"],
        fun_dim   = 2, # Expects cos/sin features
        out_dim   = 1,

        slice_num = args["slice_num"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Checkpoint] epoch={ckpt['epoch']} | "
          f"best test Rel-L2={ckpt['test_rel_l2']:.6f}")
    return model, stats


def relative_l2(pred, ref):
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-8)


# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, dataset, idx: int, device: torch.device, stats: dict):
    coords_n, az_n, file_idx = dataset[idx]
    coords_t = coords_n.unsqueeze(0).float().to(device)   # (1, N, 2)  float32
    pred_n   = model(coords_t)[0].cpu().numpy()            # (N, 1)

    # De-normalise to physical units
    az_std  = stats["az_std"]
    az_mean = stats["az_mean"]
    coord_std  = stats["coord_std"]
    coord_mean = stats["coord_mean"]
    # coords_n includes the angle features (N, 4), but physical coordinates are only the first 2
    coords_phys = coords_n[:, :2].numpy() * coord_std + coord_mean   # (N, 2)
    pred_phys   = pred_n   * az_std  + az_mean                 # (N, 1)
    true_phys   = az_n.numpy() * az_std  + az_mean             # (N, 1)

    return coords_phys, pred_phys[:, 0], true_phys[:, 0], file_idx


def plot_comparison(coords, pred, true, file_idx, save_path):
    """
    Three-panel scatter plot that reveals the actual stator geometry.

    Using scatter (not tricontourf) so that slots, teeth, and the air-gap
    appear as white space — matching the reference FEM visualisation style.
    """
    abs_err = np.abs(pred - true)
    rel_l2  = relative_l2(pred, true)
    mae     = abs_err.mean()

    x, y = coords[:, 0], coords[:, 1]

    # --- point size: small enough to show geometry gaps, large enough to
    #     fill solid regions without gaps in the yoke / teeth
    N   = len(x)
    # Heuristic: denser mesh → smaller dots
    pt  = max(0.3, min(2.0, 300_000 / N))

    # Shared Az colour limits (truth and prediction share the same scale)
    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.patch.set_facecolor('white')

    panels = [
        (true,    f"True $A_z$ Field",      'viridis',  vmin, vmax, "Az (Wb/m)"),
        (pred,    f"Predicted $A_z$ Field", 'viridis',  vmin, vmax, "Az (Wb/m)"),
        (abs_err, "Absolute Error",          'inferno',  0,    abs_err.max(), "|error| (Wb/m)"),
    ]

    for ax, (values, title, cmap, lo, hi, cbar_label) in zip(axes, panels):
        sc = ax.scatter(x, y, c=values, cmap=cmap,
                        vmin=lo, vmax=hi,
                        s=pt, linewidths=0, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Stator Angle Index = {file_idx}  |  Rel-L2 = {rel_l2:.4f}  |  "
        f"MAE = {mae:.3e} Wb/m",
        fontsize=12, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[Plot] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------
def plot_loss_curves(loss_log_path: str, save_path: str):
    """
    Read the tab-separated loss_log.txt produced by train.py and plot
    a 2-panel figure: (left) MSE vs epoch, (right) Rel-L2 vs epoch.
    Train curve is plotted every epoch; test curve only at eval epochs.
    """
    if not os.path.isfile(loss_log_path):
        print(f"[Loss curves] log not found: {loss_log_path}")
        return

    epochs_all, tr_mse, tr_rel = [], [], []
    epochs_test, te_mse, te_rel = [], [], []

    with open(loss_log_path) as fh:
        header = fh.readline()          # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ep = int(parts[0])
            epochs_all.append(ep)
            tr_mse.append(float(parts[1]))
            tr_rel.append(float(parts[2]))
            if parts[3].strip():        # test columns present
                epochs_test.append(ep)
                te_mse.append(float(parts[3]))
                te_rel.append(float(parts[4]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor('white')

    # ---- MSE panel ----
    ax1.plot(epochs_all, tr_mse, color='steelblue', lw=1.5,
             label='Train MSE', alpha=0.9)
    if epochs_test:
        ax1.plot(epochs_test, te_mse, color='darkorange', lw=2.0,
                 marker='o', markersize=3.5, label='Test MSE')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('MSE (normalised)', fontsize=10)
    ax1.set_title('Training & Test MSE', fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', ls='--', alpha=0.4)
    ax1.set_facecolor('#f9f9f9')

    # ---- Rel-L2 panel ----
    ax2.plot(epochs_all, tr_rel, color='steelblue', lw=1.5,
             label='Train Rel-L2', alpha=0.9)
    if epochs_test:
        ax2.plot(epochs_test, te_rel, color='darkorange', lw=2.0,
                 marker='o', markersize=3.5, label='Test Rel-L2')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Relative L2 Error', fontsize=10)
    ax2.set_title('Training & Test Rel-L2', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    ax2.set_facecolor('#f9f9f9')

    # Best test annotations
    if te_rel:
        best_idx = int(np.argmin(te_rel))
        ax2.annotate(
            f"Best: {te_rel[best_idx]:.4f}\n(ep {epochs_test[best_idx]})",
            xy=(epochs_test[best_idx], te_rel[best_idx]),
            xytext=(epochs_test[best_idx] + max(epochs_all)*0.05,
                    te_rel[best_idx] * 1.5),
            fontsize=8, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[Loss curves] Saved -> {save_path}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   default='checkpoints/best_model.pth')
    parser.add_argument('--gpu',          type=int, default=0)
    parser.add_argument('--results_dir',  default='results')
    parser.add_argument('--data_dir',     default=None)
    parser.add_argument('--loss_log',     default=None,
                        help="Path to loss_log.txt (default: results_dir/loss_log.txt)")
    parser.add_argument('--save_fields',  action='store_true',
                        help="Save node fields as .txt")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    n_gpu    = torch.cuda.device_count()
    use_cuda = (0 <= args.gpu < n_gpu) and torch.cuda.is_available()
    device   = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

    model, stats = load_model(args.checkpoint, device)

    data_dir = args.data_dir or DATA_DIR
    test_ds  = StatorDataset(TEST_IDXS, stats, data_dir)

    for idx in range(len(test_ds)):
        coords, pred, true, file_idx = predict(
            model, test_ds, idx, device, stats)

        rel = relative_l2(pred, true)
        mae = np.abs(pred - true).mean()
        print(f"\n[IDX={file_idx}]  Rel-L2 = {rel:.6f}  |  "
              f"MAE = {mae:.4e} Wb/m  |  N_nodes = {len(coords)}")

        # -- field prediction plot --
        plot_path = os.path.join(
            args.results_dir, f"idx{file_idx}_prediction.png")
        plot_comparison(coords, pred, true, file_idx, plot_path)

        # -- save prediction in original format (x, y, z, Ax, Ay, Az) --
        if args.save_fields:
            txt_path = os.path.join(
                args.results_dir, f"idx{file_idx}_pred_original_format.txt")
            header   = "x\ty\tz\tAx\tAy\tAz"
            zeros    = np.zeros_like(pred)
            data_out = np.column_stack([
                coords[:, 0], coords[:, 1], zeros, zeros, zeros, pred])
            np.savetxt(txt_path, data_out,
                       delimiter='\t', header=header,
                       fmt='%.16e', comments='')
            print(f"[Fields] Saved -> {txt_path}")

    # -- loss curves --
    log_path = args.loss_log or os.path.join(args.results_dir, "loss_log.txt")
    curve_path = os.path.join(args.results_dir, "loss_curves.png")
    plot_loss_curves(log_path, curve_path)


if __name__ == "__main__":
    main()
