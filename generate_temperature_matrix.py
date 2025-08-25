import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import time
from skimage import color

def parse_lut_file(lut_path):
    lut = {}
    with open(lut_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                r, g, b, temp = parts
                lut[f"{r},{g},{b}"] = float(temp)
    return lut

def map_temperatures_from_lut(image_path, lut, delta_e_threshold=25.0):
    image = Image.open(image_path).convert('RGB')
    rgb_array = np.array(image).reshape(-1, 3)

    # Debug: check RGB intersection
    unique_image_rgb = np.unique(rgb_array, axis=0)
    lut_rgb = np.array([list(map(int, k.split(','))) for k in lut.keys()])
    lut_rgb_set = set([tuple(rgb) for rgb in lut_rgb])
    image_rgb_set = set([tuple(rgb) for rgb in unique_image_rgb])
    intersection = lut_rgb_set & image_rgb_set
    print(f"[Debug] Overlapping RGB values: {len(intersection)} / {len(lut_rgb_set)} LUT keys")

    rgb_array_lab = color.rgb2lab(rgb_array.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    lut_lab = color.rgb2lab(lut_rgb.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    temperature_matrix = np.full(len(rgb_array), np.nan)

    for idx, pixel_lab in enumerate(rgb_array_lab):
        deltas = np.linalg.norm(lut_lab - pixel_lab, axis=1)
        min_delta_idx = np.argmin(deltas)

        if deltas[min_delta_idx] < delta_e_threshold:
            key = ','.join(map(str, lut_rgb[min_delta_idx]))
            temperature_matrix[idx] = lut[key]
        else:
            # Fallback to exact RGB match
            fallback_key = ','.join(map(str, rgb_array[idx]))
            if fallback_key in lut:
                temperature_matrix[idx] = lut[fallback_key]

    return temperature_matrix.reshape(np.array(image).shape[:2])

def generate_temperature_matrix_from_roi(folder_path, lut_path):
    lut = parse_lut_file(lut_path)
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".tiff"):
            image_path = os.path.join(folder_path, filename)
            print(f"\n[generate_temperature_matrix_from_roi] Processing {filename}")
            start = time.time()

            temperature_matrix = map_temperatures_from_lut(image_path, lut)

            valid = np.count_nonzero(~np.isnan(temperature_matrix))
            total = temperature_matrix.size
            print(f"[generate_temperature_matrix_from_roi] Matched {valid}/{total} pixels with ΔE < 25.0")
            print(f"[generate_temperature_matrix_from_roi] Execution time: {time.time() - start:.4f} seconds")

            save_name = filename.replace('.tiff', '_temp_matrix.npy')
            np.save(os.path.join(folder_path, save_name), temperature_matrix)

            visualize_temperature_matrix(temperature_matrix, filename)

def visualize_temperature_matrix(temp_matrix, title):
    if np.isnan(temp_matrix).all():
        print("[visualize_temperature_matrix] All values are NaN. Skipping heatmap.")
        return

    plt.figure(figsize=(10, 8))
    masked = np.ma.masked_invalid(temp_matrix)
    cmap = plt.get_cmap('jet')
    norm = Normalize(vmin=np.nanmin(temp_matrix), vmax=np.nanmax(temp_matrix))
    plt.imshow(masked, cmap=cmap, norm=norm)
    plt.title(f"Temperature Matrix - {title}")
    plt.colorbar(label="Temperature (°C)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    folder_path = "output/super_resolved"
    lut_path = "output/lut_mapping.csv"
    generate_temperature_matrix_from_roi(folder_path, lut_path)
