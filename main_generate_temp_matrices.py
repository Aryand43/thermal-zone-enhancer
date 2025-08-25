import os
import numpy as np
from generate_temperature_matrix import generate_temperature_matrix_from_roi
from generate_temperature_matrix import visualize_temperature_matrix  

input_dir = "checkpoint_1_detect_and_crop"
output_dir = "temperature_matrices"
plot_output_dir = "temperature_heatmaps"
lut_path = "color_temp_lut.json"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

for i in range(1, 90):
    image_path = f"{input_dir}/{i}.tiff"
    save_path = f"{output_dir}/{i}.npy"
    heatmap_path = f"{plot_output_dir}/{i}.png"

    try:
        temp_matrix = generate_temperature_matrix_from_roi(image_path, lut_path, save_path)
        visualize_temperature_matrix(temp_matrix, title=f"Frame {i} Temperature Heatmap", save_path=heatmap_path)
    except Exception as e:
        print(f"[ERROR] Failed on {i}.tiff: {e}")
