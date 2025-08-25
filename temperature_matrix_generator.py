import cv2
import numpy as np
import json
import os
import time
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.pyplot as plt

def map_temperatures_from_lut(image_path: str, lut_path: str, save_path: str, delta_e_threshold: float = 35.0) -> np.ndarray:
    """
    Maps per-pixel temperatures from thermal imagery using RGB-to-temperature LUT
    with ΔE-CIEDE2000 precision filtering.

    Matches: Step 7 (concurrent) in invention. Produces °C matrix aligned with enhanced image.
    Novelty: Thermal-only mapping with ΔE-based LAB color distance filtering.
    """
    start_time = time.perf_counter()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(lut_path, 'r') as f:
        lut = json.load(f)

    # Prepare image and LUT in LAB color space
    height, width, _ = image_rgb.shape
    rgb_array = image_rgb.reshape(-1, 3).astype(np.uint8)
    temperature_matrix = np.full((height * width,), np.nan, dtype=np.float32)

    unique_rgb = np.array([list(map(int, key.split(','))) for key in lut.keys()])
    unique_lab = rgb2lab(unique_rgb[np.newaxis, :, :]).reshape(-1, 3)
    rgb_lab = rgb2lab(rgb_array[np.newaxis, :, :]).reshape(-1, 3)

    match_count = 0
    for idx, pixel_lab in enumerate(rgb_lab):
        deltas = deltaE_ciede2000(np.tile(pixel_lab, (unique_lab.shape[0], 1)), unique_lab)
        min_delta_idx = np.argmin(deltas)
        if deltas[min_delta_idx] < delta_e_threshold:
            key = ','.join(map(str, unique_rgb[min_delta_idx]))
            temperature_matrix[idx] = lut[key]
            match_count += 1

    print(f"[generate_temperature_matrix_from_roi] Matched {match_count}/{len(rgb_lab)} pixels with ΔE < {delta_e_threshold}")

    temperature_matrix = temperature_matrix.reshape((height, width))
    np.save(save_path, temperature_matrix)

    end_time = time.perf_counter()
    print(f"[generate_temperature_matrix_from_roi] Execution time: {end_time - start_time:.4f} seconds")
    return temperature_matrix

def visualize_temperature_matrix(matrix: np.ndarray, title: str = "Temperature Heatmap", save_path: str = None):
    if np.isnan(matrix).all():
        print("[visualize_temperature_matrix] All values are NaN. Skipping heatmap.")
        return

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Temperature (°C")
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
