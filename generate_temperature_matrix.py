import cv2
import numpy as np
import json
import os
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def generate_temperature_matrix_from_roi(image_path: str, lut_path: str, save_path: str) -> np.ndarray:
    start_time = time.perf_counter()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(lut_path, 'r') as f:
        lut = json.load(f)

    rgb_values = []
    temperatures = []
    for key, temp in lut.items():
        r, g, b = map(int, key.split(','))
        rgb_values.append([r, g, b])
        temperatures.append(temp)

    kdtree = KDTree(rgb_values)
    height, width, _ = image_rgb.shape
    temperature_matrix = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r, g, b = image_rgb[i, j]
            distance, index = kdtree.query([r, g, b])
            temperature_matrix[i, j] = temperatures[index]

    np.save(save_path, temperature_matrix)

    end_time = time.perf_counter()
    print(f"[generate_temperature_matrix_from_roi] Execution time: {end_time - start_time:.4f} seconds")
    return temperature_matrix

def visualize_temperature_matrix(matrix: np.ndarray, title: str = "Temperature Heatmap", save_path: str = None):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Temperature (°C)")
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()