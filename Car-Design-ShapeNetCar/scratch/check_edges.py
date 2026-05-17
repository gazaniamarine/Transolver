import torch
import sys
import os

# Set up paths
repo_root = '/home/gazania/zania_folder/Transolver/Car-Design-ShapeNetCar'
sys.path.append(repo_root)

from dataset.load_dataset import load_train_val_fold
from dataset.dataset import create_edge_index_radius

# Mock args
class Args:
    data_dir = os.path.join(repo_root, 'dataset/mlcfd_data/training_data')
    save_dir = os.path.join(repo_root, 'dataset/mlcfd_data/preprocessed_data')
    fold_id = 0
    cfd_mesh = False
    r = 0.2

args = Args()

# Note: get_datalist is imported from dataset.dataset inside load_dataset
# which works because we added repo_root to sys.path

train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=1)
sample = train_data[0]
print(f"Sample loaded from preprocessed data.")
print(f"Original edge_index shape (from file): {sample.edge_index.shape}")

sample_radius = create_edge_index_radius(sample.clone(), r=0.2)
print(f"Radius graph (r=0.2) edge_index shape: {sample_radius.edge_index.shape}")
