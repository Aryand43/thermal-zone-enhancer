import cv2
import numpy as np
import json
import os
import time
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.pyplot as plt

def map_temperatures_from_lut(
    image_path: str,
    lut_path: str,
    save_path: str,
    delta_e_threshold: float = 35.0,
    fallback_rgb_match: bool = True,
    report_coverage: bool = True,
    return_nan_mask: bool = False
) -> np.ndarray:
    start_time = time.perf_counter()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(lut_path, 'r') as f:
        lut = json.load(f)

    height, width, _ = image_rgb.shape
    rgb_array = image_rgb.reshape(-1, 3).astype(np.uint8)
    temperature_matrix = np.full((height * width,), np.nan, dtype=np.float32)

    unique_rgb = np.array([list(map(int, key.split(','))) for key in lut.keys()])
    unique_lab = rgb2lab(unique_rgb[np.newaxis, :, :]).reshape(-1, 3)
    rgb_lab = rgb2lab(rgb_array[np.newaxis, :, :]).reshape(-1, 3)

    match_count = 0
    fallback_count = 0

    for idx, pixel_lab in enumerate(rgb_lab):
        deltas = deltaE_ciede2000(np.tile(pixel_lab, (unique_lab.shape[0], 1)), unique_lab)
        min_idx = np.argmin(deltas)
        if deltas[min_idx] < delta_e_threshold:
            key = ','.join(map(str, unique_rgb[min_idx]))
            temperature_matrix[idx] = lut[key]
            match_count += 1
        elif fallback_rgb_match:
            rgb_key = ','.join(map(str, rgb_array[idx]))
            if rgb_key in lut:
                temperature_matrix[idx] = lut[rgb_key]
                fallback_count += 1

    coverage = (np.count_nonzero(~np.isnan(temperature_matrix)) / len(temperature_matrix)) * 100
    temperature_matrix = temperature_matrix.reshape((height, width))
    np.save(save_path, temperature_matrix)

    if report_coverage:
        print(f"[ΔE LUT Mapping] Matched: {match_count}, Fallback: {fallback_count}, Coverage: {coverage:.2f}%")
    print(f"[ΔE LUT Mapping] Saved: {save_path}")
    print(f"[ΔE LUT Mapping] Time: {time.perf_counter() - start_time:.2f}s")

    if return_nan_mask:
        return temperature_matrix, np.isnan(temperature_matrix).astype(np.uint8)
    return temperature_matrix

def visualize_temperature_matrix(matrix: np.ndarray, title: str = "Temperature Heatmap", save_path: str = None):
    if np.isnan(matrix).all():
        print("[visualize_temperature_matrix] All values are NaN. Skipping heatmap.")
        return

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Temperature (°C)")
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
