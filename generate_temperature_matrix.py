import cv2
import numpy as np
import json
import os
import time

def generate_temperature_matrix_from_roi(image_path: str, lut_path: str, save_path: str) -> np.ndarray:
    start_time = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(lut_path, 'r') as f:
        lut = json.load(f)
    height, width, _ = image_rgb.shape
    temperature_matrix = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            r, g, b = image_rgb[i, j]
            rgb_key = f"{r},{g},{b}"
            temperature = lut.get(rgb_key, 0.0)  # Default to 0.0 if not found
            temperature_matrix[i, j] = temperature
    np.save(save_path, temperature_matrix)
    end_time = time.perf_counter()
    print(f"[generate_temperature_matrix_from_roi] Execution time: {end_time - start_time:.4f} seconds")
    return temperature_matrix
