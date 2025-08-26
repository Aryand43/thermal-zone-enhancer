import os
import numpy as np
from temperature_matrix_generator import map_temperatures_from_lut, visualize_temperature_matrix
INPUT_DIR = "checkpoint_1_detect_and_crop"
OUTPUT_DIR = "temperature_matrices"
PLOT_OUTPUT_DIR = "temperature_heatmaps"
LUT_PATH = "color_temp_lut.json"
DELTA_E_THRESHOLD = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

for i in range(1, 90):
    image_path = os.path.join(INPUT_DIR, f"{i}.tiff")
    save_path = os.path.join(OUTPUT_DIR, f"{i}.npy")
    heatmap_path = os.path.join(PLOT_OUTPUT_DIR, f"{i}.png")

    try:
        temp_matrix = map_temperatures_from_lut(
            image_path=image_path,
            lut_path=LUT_PATH,
            save_path=save_path,
            delta_e_threshold=DELTA_E_THRESHOLD,
            fallback_rgb_match=True,
            report_coverage=True,
            return_nan_mask=False
        )
        visualize_temperature_matrix(temp_matrix, title=f"Frame {i} Temperature Heatmap", save_path=heatmap_path)
    except Exception as e:
        print(f"[ERROR] Failed on {i}.tiff: {e}")
