import os
import numpy as np
from thermal_gradient_analysis import (
    track_melt_pool_boundary_and_gradient,
    visualize_hot_pixels,
    calculate_pixel_velocities,
    calculate_weighted_velocities,
    plot_velocity_time_graph
)
from config_material import MATERIAL_CONFIG
if __name__ == "__main__":
    # Config
    temp_matrix_dir = "temperature_matrices"
    pixel_resolution_um = MATERIAL_CONFIG["pixel_resolution_um"]
    pixel_resolution_m = pixel_resolution_um * 1e-6
    frame_rate = MATERIAL_CONFIG["frame_rate"]
    time_interval = 1.0 / frame_rate

    # Output
    output_dir = "thermal_gradient_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("==> Tracking melt pool boundary and gradients...")
    hot_pixel_coords, position_shifts, thermal_gradients = track_melt_pool_boundary_and_gradient(
        temp_dir=temp_matrix_dir,
        pixel_resolution=pixel_resolution_m
    )

    np.save(f"{output_dir}/hot_pixel_coords.npy", np.array(hot_pixel_coords, dtype=object))
    np.save(f"{output_dir}/position_shifts.npy", np.array(position_shifts, dtype=object))
    np.save(f"{output_dir}/thermal_gradients.npy", np.array(thermal_gradients, dtype=object))
    print(f"[✓] Saved gradient arrays to '{output_dir}'")

    print("==> Visualizing hot pixel frames...")
    visualize_hot_pixels(
        hot_pixel_coords_file=f"{output_dir}/hot_pixel_coords.npy",
        output_dir=f"{output_dir}/hot_pixel_frames"
    )

    print("==> Plotting average unweighted velocity...")
    unweighted = calculate_pixel_velocities(position_shifts, pixel_resolution_um, time_interval)
    np.save(f"{output_dir}/velocity_unweighted.npy", np.array(unweighted, dtype=object))
    plot_velocity_time_graph(unweighted, output_path=f"{output_dir}/velocity_avg_unweighted.png")

    print("==> Plotting weighted velocity (1385–1450 °C)...")
    weighted_1385 = calculate_weighted_velocities(
        temp_dir=temp_matrix_dir,
        position_shifts=position_shifts,
        temp_range=(1000, 1800),
        pixel_resolution=pixel_resolution_um,
        time_step=time_interval
    )
    np.save(f"{output_dir}/velocity_weighted_1385_1450.npy", np.array(weighted_1385))
    plot_velocity_time_graph(weighted_1385, output_path=f"{output_dir}/velocity_weighted_1385_1450.png")

    print("==> Plotting weighted velocity (>1600 °C)...")
    weighted_1600 = calculate_weighted_velocities(
        temp_dir=temp_matrix_dir,
        position_shifts=position_shifts,
        temp_range=(1600, float('inf')),
        pixel_resolution=pixel_resolution_um,
        time_step=time_interval
    )
    np.save(f"{output_dir}/velocity_weighted_gt1600.npy", np.array(weighted_1600))
    plot_velocity_time_graph(weighted_1600, output_path=f"{output_dir}/velocity_weighted_gt1600.png")

    print("==> DONE. Check output folder:", output_dir)
