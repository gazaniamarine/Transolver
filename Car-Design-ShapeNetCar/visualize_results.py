import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset

def visualize_sample(model, device, dataset, coef_norm, sample_idx=0, save_path="prediction_vis.png"):
    model.eval()
    
    # Get a sample
    cfd_data, geom = dataset[sample_idx]
    
    # Apply normalization if provided
    if coef_norm:
        mean_in, std_in, mean_out, std_out = coef_norm
        cfd_data.x = ((cfd_data.x - torch.tensor(mean_in)) / (torch.tensor(std_in) + 1e-8)).float()
        # We don't normalize y here because we want to compare with true values later, 
        # but the model will predict normalized values if it was trained that way.
    
    cfd_data = cfd_data.to(device)
    cfd_data.x = cfd_data.x.float()
    cfd_data.pos = cfd_data.pos.float()
    
    geom = geom.unsqueeze(0).to(device).float()
    
    # Run inference
    with torch.no_grad():
        from torch_geometric.data import Batch
        cfd_batch = Batch.from_data_list([cfd_data])
        
        out = model((cfd_batch, geom))
        
        # If model expects normalized output, we must denormalize
        if coef_norm:
            mean_out = torch.tensor(coef_norm[2]).to(device)
            std_out = torch.tensor(coef_norm[3]).to(device)
            out = out * std_out + mean_out
            
        out = out.cpu().numpy()
        targets = cfd_data.y.cpu().numpy()
        pos = cfd_data.pos.cpu().numpy()
        surf_mask = cfd_data.surf.cpu().numpy()

    # Split output: [velo_x, velo_y, velo_z, pressure]
    pred_velo = out[:, :3]
    pred_press = out[:, 3]
    
    true_velo = targets[:, :3]
    true_press = targets[:, 3]

    # Calculate velocity magnitude
    pred_velo_mag = np.linalg.norm(pred_velo, axis=1)
    true_velo_mag = np.linalg.norm(true_velo, axis=1)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), subplot_kw={'projection': '3d'})
    
    # Plotting surface points only for better visibility
    surf_pos = pos[surf_mask]
    surf_pred_press = pred_press[surf_mask]
    surf_true_press = true_press[surf_mask]
    surf_press_err = np.abs(surf_pred_press - surf_true_press)
    
    # ROW 1: PRESSURE
    # 1. True Pressure
    sc1 = axes[0, 0].scatter(surf_pos[:, 0], surf_pos[:, 1], surf_pos[:, 2], c=surf_true_press, cmap='jet', s=1)
    axes[0, 0].set_title('True Pressure (Surface)')
    fig.colorbar(sc1, ax=axes[0, 0], label='Pressure')
    
    # 2. Predicted Pressure
    sc2 = axes[0, 1].scatter(surf_pos[:, 0], surf_pos[:, 1], surf_pos[:, 2], c=surf_pred_press, cmap='jet', s=1)
    axes[0, 1].set_title('Predicted Pressure (Surface)')
    fig.colorbar(sc2, ax=axes[0, 1], label='Pressure')

    # 3. Pressure Error
    sc3 = axes[0, 2].scatter(surf_pos[:, 0], surf_pos[:, 1], surf_pos[:, 2], c=surf_press_err, cmap='magma', s=1)
    axes[0, 2].set_title('Pressure Absolute Error')
    fig.colorbar(sc3, ax=axes[0, 2], label='Error')

    # ROW 2: VELOCITY
    step = 5
    # 4. True Velocity Magnitude
    sc4 = axes[1, 0].scatter(pos[::step, 0], pos[::step, 1], pos[::step, 2], c=true_velo_mag[::step], cmap='viridis', s=0.5)
    axes[1, 0].set_title('True Velocity Magnitude')
    fig.colorbar(sc4, ax=axes[1, 0], label='Velocity')

    # 5. Predicted Velocity Magnitude
    sc5 = axes[1, 1].scatter(pos[::step, 0], pos[::step, 1], pos[::step, 2], c=pred_velo_mag[::step], cmap='viridis', s=0.5)
    axes[1, 1].set_title('Predicted Velocity Magnitude')
    fig.colorbar(sc5, ax=axes[1, 1], label='Velocity')

    # 6. Velocity Error
    velo_err = np.abs(pred_velo_mag - true_velo_mag)
    sc6 = axes[1, 2].scatter(pos[::step, 0], pos[::step, 1], pos[::step, 2], c=velo_err[::step], cmap='magma', s=0.5)
    axes[1, 2].set_title('Velocity Magnitude Absolute Error')
    fig.colorbar(sc6, ax=axes[1, 2], label='Error')

    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Standardize view
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='Car-Design-ShapeNetCar/dataset/mlcfd_data/training_data')
    parser.add_argument('--save_dir', default='Car-Design-ShapeNetCar/dataset/mlcfd_data/preprocessed_data')
    parser.add_argument('--model_path', default='metrics/Transolver/0/200_0.5/model_200.pth')
    parser.add_argument('--fold_id', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--idx', default=0, type=int, help='Sample index in validation set')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("Loading samples...")
    # Get just the validation fold samples list
    from dataset.load_dataset import get_samples
    all_samples = get_samples(args.data_dir)
    val_samples = all_samples[args.fold_id]
    
    selected_sample = [val_samples[args.idx]]
    print(f"Loading sample: {selected_sample[0]}")
    
    from dataset.load_dataset import get_datalist
    # Load just one sample - preprocessed=False to do calculations on the fly
    val_data = get_datalist(args.data_dir, selected_sample, norm=False, preprocessed=False)
    
    val_ds = GraphDataset(val_data, use_cfd_mesh=False, r=0.2)
    
    print(f"Loading model from {args.model_path}...")
    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    
    # Load normalization from log
    log_path = args.model_path.replace('model_200.pth', 'log_200.json')
    coef_norm = None
    if os.path.exists(log_path):
        print(f"Loading normalization from {log_path}...")
        import json
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            coef_norm = log_data.get('coef_norm')
    
    visualize_sample(model, device, val_ds, coef_norm, sample_idx=0) # Index is 0 because val_ds only has 1 sample
