"""
Generate Mock AirfRANS Dataset (Using Plane Grid)
================================================
Generates a small, synthetic, physically-consistent mock AirfRANS dataset
where the internal volume mesh is a quadrilateral Plane grid, allowing
successful reshaping of cells into (-1, 5).

This enables immediate, end-to-end verification of the Airfoil RANS pipeline.
"""

import os
import json
import numpy as np
import pyvista as pv

MOCK_DIR = os.environ.get('AIRFRANS_DATA_DIR', os.path.join(os.path.dirname(__file__), 'mock_dataset'))
os.makedirs(MOCK_DIR, exist_ok=True)

geometries = [
    "mock_geometry_1_10.0_5.0",
    "mock_geometry_2_12.0_2.0",
    "mock_geometry_3_15.0_-1.0",
    "mock_geometry_4_8.0_4.0",
    "mock_geometry_5_11.0_3.0"
]

print("[Mock Data] Generating synthetic AirfRANS geometries ...")

for s in geometries:
    geom_dir = os.path.join(MOCK_DIR, s)
    os.makedirs(geom_dir, exist_ok=True)

    # 1. Create Internal Volume Mesh (VTU) using pv.Plane (quadrilateral cells)
    # Rectangle on [-1, 2] x [-1, 1] -> size 3.0 x 2.0, centered at (0.5, 0.0)
    internal = pv.Plane(
        center=(0.5, 0.0, 0.0),
        direction=(0, 0, 1),
        i_size=3.0,
        j_size=2.0,
        i_resolution=14,
        j_resolution=14
    ).cast_to_unstructured_grid()
    
    # Compute Area of the 2D cells
    internal = internal.compute_cell_sizes(length=False, area=True, volume=False)
    
    n_points = internal.n_points
    
    # Parse Reynolds/Uinf and AoA from name (s)
    parts = s.split('_')
    Uinf = float(parts[-2])
    alpha = float(parts[-1]) * np.pi / 180.0
    
    # Set up physically reasonable mock fields
    center = np.array([0.5, 0.0, 0.0])
    dist = np.linalg.norm(internal.points - center, axis=1)
    internal.point_data['implicit_distance'] = dist - 0.2
    
    # U (Velocity field): free-stream velocity with deficit
    U_free = np.array([np.cos(alpha) * Uinf, np.sin(alpha) * Uinf, 0.0])
    U_field = np.tile(U_free, (n_points, 1))
    
    deficit_mask = dist < 0.4
    U_field[deficit_mask] *= (dist[deficit_mask] / 0.4)[:, None]
    internal.point_data['U'] = U_field
    
    # p (Pressure field)
    p_field = 0.5 * (Uinf**2 - np.linalg.norm(U_field, axis=1)**2)
    internal.point_data['p'] = p_field
    
    # nut (Turbulent viscosity)
    nut_field = np.maximum(0.0, 0.1 * Uinf * (0.4 - dist))
    internal.point_data['nut'] = nut_field
    
    # Save Internal VTU file
    vtu_file = os.path.join(geom_dir, f"{s}_internal.vtu")
    internal.save(vtu_file)
    print(f"  Saved internal grid → {vtu_file} (points={n_points})")

    # 2. Create Surface Mesh (VTP)
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    surf_points = np.stack([0.5 + 0.2*np.cos(theta), 0.2*np.sin(theta), np.zeros_like(theta)], axis=-1)
    
    # Connect points into a closed polyline / lines
    cells = np.zeros(30 * 3, dtype=int)
    for i in range(30):
        cells[3*i] = 2
        cells[3*i+1] = i
        cells[3*i+2] = (i + 1) % 30
        
    aerofoil = pv.PolyData(surf_points, lines=cells)
    
    # Compute line lengths
    aerofoil = aerofoil.compute_cell_sizes(length=True, area=False, volume=False)
    
    n_surf = aerofoil.n_points
    aerofoil.point_data['implicit_distance'] = np.zeros(n_surf)
    
    # Normals
    normals_3d = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    aerofoil.point_data['Normals'] = normals_3d
    
    # Velocity on wall: No-slip boundary condition (U = 0)
    aerofoil.point_data['U'] = np.zeros((n_surf, 3))
    
    # Pressure and turbulent viscosity on surface wall
    aerofoil.point_data['p'] = 0.5 * (Uinf**2) * np.ones(n_surf)
    aerofoil.point_data['nut'] = np.zeros(n_surf)
    
    # Save Surface VTP file
    vtp_file = os.path.join(geom_dir, f"{s}_aerofoil.vtp")
    aerofoil.save(vtp_file)
    print(f"  Saved airfoil boundary → {vtp_file} (points={n_surf})")

# 3. Create manifest.json
manifest = {
    "full_train": geometries[:-1],
    "full_test": [geometries[-1]]
}

manifest_file = os.path.join(MOCK_DIR, "manifest.json")
with open(manifest_file, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"\n[Mock Data] Successfully created manifest → {manifest_file}")
