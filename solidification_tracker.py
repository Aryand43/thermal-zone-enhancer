import os
import numpy as np
from typing import List, Tuple
from scipy.ndimage import center_of_mass

def track_solidification_fronts(
    temperature_matrices_dir: str,
    threshold: float = 1450.0,
    pixel_resolution_um: float = 80.0,
    frame_rate: float = 80.0
) -> List[Tuple[float, float]]:
    """
    Tracks the solidification front by identifying molten pixels (>= threshold)
    that solidify (< threshold) in the next frame.

    Returns list of velocities in micrometers per second.
    """
    files = sorted([f for f in os.listdir(temperature_matrices_dir) if f.endswith(".npy")])
    solidification_velocities = []

    for i in range(len(files) - 1):
        curr = np.load(os.path.join(temperature_matrices_dir, files[i]))
        next_ = np.load(os.path.join(temperature_matrices_dir, files[i + 1]))

        molten_mask = curr >= threshold
        solidified_mask = (curr >= threshold) & (next_ < threshold)

        if not np.any(solidified_mask):
            solidification_velocities.append((0.0, 0.0))
            continue

        # Compute centroid of solidification front
        cy, cx = center_of_mass(solidified_mask)

        # Optional: compute previous centroid too for molten boundary tracking
        if np.any(molten_mask):
            cy_prev, cx_prev = center_of_mass(molten_mask)
        else:
            cy_prev, cx_prev = cy, cx

        dx = (cx - cx_prev) * pixel_resolution_um  # μm
        dy = (cy - cy_prev) * pixel_resolution_um
        dt = 1.0 / frame_rate  # seconds

        vx = dx / dt
        vy = dy / dt

        solidification_velocities.append((vx, vy))

    return solidification_velocities
