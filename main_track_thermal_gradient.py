from thermal_gradient_tracker import (
    track_melt_pool_boundary_and_gradient,
    visualize_hot_pixels,
    calculate_pixel_velocities,
    calculate_weighted_velocities,
    plot_velocity_time_graph
)

import numpy as np
import os

if __name__ == "__main__":
    # Configuration
    temp_matrix_dir = "temperature_matrices"
    pixel_resolution_um = 31.3  # micrometers per pixel
    pixel_resolution_m = pixel_resolution_um * 1e-6  # convert to meters for tracking
    time_interval = 0.0125  # seconds per frame

    # Step 1: Track melt pool and gradients
    hot_pixel_coords, position_shifts, thermal_gradients = track_melt_pool_boundary_and_gradient(
        temp_dir=temp_matrix_dir,
        pixel_resolution=pixel_resolution_m
    )

    # Output directory setup
    os.makedirs("thermal_gradient_outputs", exist_ok=True)

    # Step 2: Save arrays
    np.save("thermal_gradient_outputs/hot_pixel_coords.npy", np.array(hot_pixel_coords, dtype=object))
    np.save("thermal_gradient_outputs/position_shifts.npy", np.array(position_shifts, dtype=object))
    np.save("thermal_gradient_outputs/thermal_gradients.npy", np.array(thermal_gradients, dtype=object))

    print("Tracking complete.")
    print(f"Frames processed: {len(position_shifts)}")
    print("Data saved to 'thermal_gradient_outputs/'")

    # Step 3: Visualize hot pixels
    visualize_hot_pixels(
        hot_pixel_coords_file="thermal_gradient_outputs/hot_pixel_coords.npy",
        output_dir="thermal_gradient_outputs/hot_pixel_frames"
    )

    # Step 4: Normal velocity calculation
    pixel_velocities = calculate_pixel_velocities(
        position_shifts=position_shifts,
        pixel_resolution=pixel_resolution_um,  # micrometers
        time_step=time_interval
    )

    plot_velocity_time_graph(
        pixel_velocities=pixel_velocities,
        output_path="thermal_gradient_outputs/velocity_plot_unweighted.png"
    )

    # Step 5: Weighted velocity for pixels in 1385–1450°C
    weighted_velocities = calculate_weighted_velocities(
        temp_dir=temp_matrix_dir,
        position_shifts=position_shifts,
        temp_range=(1385, 1450),
        pixel_resolution=pixel_resolution_um,
        time_step=time_interval
    )

    plot_velocity_time_graph(
        pixel_velocities=weighted_velocities,
        output_path="thermal_gradient_outputs/velocity_plot_weighted_1385_1450.png"
    )

    # Optional: Add another weighted plot for >1600°C
    weighted_velocities_1600 = calculate_weighted_velocities(
        temp_dir=temp_matrix_dir,
        position_shifts=position_shifts,
        temp_range=(1600, float('inf')),
        pixel_resolution=pixel_resolution_um,
        time_step=time_interval
    )

    plot_velocity_time_graph(
        pixel_velocities=weighted_velocities_1600,
        output_path="thermal_gradient_outputs/velocity_plot_weighted_gt1600.png"
    )
