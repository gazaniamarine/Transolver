import numpy as np

def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n, dtype=int)
    for i in range(n):
        # Find index of closest point in in_order_points
        diffs = in_order_points - out_order_points[i]
        dists = np.sum(diffs ** 2, axis=1)
        idx[i] = np.argmin(dists)

    return quantity_to_reordered[idx]