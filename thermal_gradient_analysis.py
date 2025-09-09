import numpy as np
import os
import matplotlib.pyplot as plt
from config_material import MATERIAL_CONFIG

def track_melt_pool_boundary_and_gradient(temp_dir: str, pixel_resolution: float = 31.3e-6):
    file_names = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npy')],
                        key=lambda x: int(os.path.splitext(x)[0]))

    hot_pixel_coords = []
    position_shifts = []
    thermal_gradients = []

    for file in file_names:
        temp_matrix = np.load(os.path.join(temp_dir, file))
        liquidus = MATERIAL_CONFIG["liquidus_temp"]
        solidus = MATERIAL_CONFIG["solidus_temp"]

        coords = np.argwhere((temp_matrix >= solidus) & (temp_matrix <= liquidus))        
        hot_pixel_coords.append(coords)

    for i in range(1, len(hot_pixel_coords)):
        prev_coords = hot_pixel_coords[i - 1]
        curr_coords = hot_pixel_coords[i]

        if prev_coords.size == 0 or curr_coords.size == 0:
            position_shifts.append([])
            thermal_gradients.append([])
            print(f"[Frame {i}] No valid gradients or displacements.")
            continue

        prev_rows = set(tuple(r) for r in prev_coords[:, 0:1])
        curr_rows = set(tuple(r) for r in curr_coords[:, 0:1])
        common_rows = np.array(list(prev_rows & curr_rows)).flatten()

        dxs = []
        grads = []

        for row in common_rows:
            prev_cols = prev_coords[prev_coords[:, 0] == row][:, 1]
            curr_cols = curr_coords[curr_coords[:, 0] == row][:, 1]

            if len(prev_cols) == 0 or len(curr_cols) == 0:
                continue

            prev_mean_col = np.mean(prev_cols)
            curr_mean_col = np.mean(curr_cols)

            dx = abs(curr_mean_col - prev_mean_col)
            if dx == 0:
                continue

            try:
                prev_temp = np.load(os.path.join(temp_dir, file_names[i - 1]))[row, int(prev_mean_col)]
                curr_temp = np.load(os.path.join(temp_dir, file_names[i]))[row, int(curr_mean_col)]
                dT = curr_temp - prev_temp
                grad = dT / (dx * pixel_resolution + 1e-8)
            except:
                continue

            dxs.append(dx)
            grads.append(grad)

        if not dxs or not grads:
            position_shifts.append([])
            thermal_gradients.append([])
            print(f"[Frame {i}] No valid gradients or displacements.")
            continue

        position_shifts.append(dxs)
        thermal_gradients.append(grads)

        avg_dx = np.mean(dxs)
        avg_grad = np.mean(grads)
        print(f"[Frame {i}] Avg dx: {avg_dx:.2f} px | Avg grad: {avg_grad:.2f} °C/m")

    return hot_pixel_coords, position_shifts, thermal_gradients


def calculate_pixel_velocities(position_shifts, pixel_resolution=31.3, time_step=0.0125):
    pixel_velocities = []
    for shifts in position_shifts:
        if not shifts:
            pixel_velocities.append([])
            continue
        velocities = [(abs(dx) * pixel_resolution) / time_step for dx in shifts]
        pixel_velocities.append(velocities)
    return pixel_velocities

def calculate_weighted_velocities(temp_dir, position_shifts, temp_range, pixel_resolution, time_step):
    velocity_per_frame = []
    temp_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npy')],
                        key=lambda x: int(os.path.splitext(x)[0]))

    for idx, temp_file in enumerate(temp_files[1:], start=1):
        try:
            curr_temp_matrix = np.load(os.path.join(temp_dir, temp_file))
        except Exception:
            velocity_per_frame.append(0)
            continue

        shifts = position_shifts[idx - 1]
        if not isinstance(shifts, (tuple, list)) or len(shifts) == 0:
            velocity_per_frame.append(0)
            continue

        mean_dx = np.mean([abs(dx) for dx in shifts])
        if np.isnan(mean_dx) or np.isinf(mean_dx):
            velocity_per_frame.append(0)
            continue

        mask = (curr_temp_matrix >= temp_range[0]) & (curr_temp_matrix <= temp_range[1])
        weight = np.sum(mask)

        if weight == 0:
            velocity_per_frame.append(0)
            continue

        displacement_um = mean_dx * pixel_resolution
        velocity = (displacement_um / time_step) * weight
        velocity_per_frame.append(velocity)

    return velocity_per_frame

def plot_velocity_time_graph(pixel_velocities, output_path="thermal_gradient_outputs/velocity_plot.png"):
    # Handle flat list (weighted) vs nested list (unweighted)
    if isinstance(pixel_velocities[0], (int, float, np.number)):
        avg_velocities = pixel_velocities
    else:
        avg_velocities = [np.mean(v) if len(v) > 0 else 0 for v in pixel_velocities]

    time_points = [i * 0.0125 for i in range(len(avg_velocities))]

    plt.figure(figsize=(8, 5))
    plt.plot(time_points, avg_velocities, marker='o', color='blue')
    plt.title("Average Pixel Velocity Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (μm/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Velocity graph saved to '{output_path}'.")

def visualize_hot_pixels(hot_pixel_coords_file: str, output_dir: str):
    if not os.path.exists(hot_pixel_coords_file):
        print(f"Error: File '{hot_pixel_coords_file}' not found.")
        return

    try:
        hot_pixel_coords = np.load(hot_pixel_coords_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if len(hot_pixel_coords) == 0:
        print("Warning: The data file is empty. No plots to generate.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for idx, frame_coords in enumerate(hot_pixel_coords):
        if frame_coords.size == 0:
            print(f"Skipping frame {idx} (no hot pixels).")
            continue

        y, x = frame_coords[:, 0], frame_coords[:, 1]

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=10, c='red')
        plt.title(f"Frame {idx} – Hot Pixels (1385°C to 1450°C)")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"frame_{idx}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Visualization complete. Plots saved in '{output_dir}'.")
