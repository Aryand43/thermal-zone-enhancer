import numpy as np
import os
import time
import json
from PIL import Image 

def build_color_temp_lut(ref_image, temp_csv_path):
    start_time = time.perf_counter()
    if not isinstance(ref_image, np.ndarray):
        raise TypeError("ref_image must be a numpy ndarray.")
    if ref_image.ndim != 3 or ref_image.shape[2] != 3:
        raise ValueError("ref_image must be an RGB image with shape (H, W, 3).")
    if not isinstance(temp_csv_path, str) or not os.path.isfile(temp_csv_path):
        raise FileNotFoundError(f"CSV file not found: {temp_csv_path}")
    temp_mat = np.genfromtxt(temp_csv_path, delimiter=",")
    if temp_mat.shape[1] > ref_image.shape[1]:
        temp_mat = temp_mat[:, :ref_image.shape[1]]
    if temp_mat.ndim != 2:
        raise ValueError("Temperature CSV must contain a 2D matrix.")
    H, W, _ = ref_image.shape
    if temp_mat.shape != (H, W):
        raise ValueError(f"Temperature matrix shape {temp_mat.shape} does not match image spatial dimensions {(H, W)}.")
    rgb = ref_image.reshape(-1, 3)
    temps = temp_mat.reshape(-1).astype(float)
    if not np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.int64)
    unique_rgb, inv = np.unique(rgb, axis=0, return_inverse=True)
    temp_sum = np.bincount(inv, weights=temps)
    temp_count = np.bincount(inv)
    avg_temp = temp_sum / temp_count
    lut = {tuple(color.tolist()): float(avg_temp[i]) for i, color in enumerate(unique_rgb)}
    elapsed = time.perf_counter() - start_time
    print(f"build_color_temp_lut latency: {elapsed:.6f} s")
    return lut

def main():
    image_path = "Temp(1).tiff"
    temp_csv_path = "Temperature(1).csv"
    output_json_path = "color_temp_lut.json"
    ref_image = np.array(Image.open(image_path))
    lut = build_color_temp_lut(ref_image, temp_csv_path)
    with open(output_json_path, "w") as f:
        lut_str_keys = {f"{r},{g},{b}": temp for (r, g, b), temp in lut.items()}
        json.dump(lut_str_keys, f)
    print(f"Saved LUT with {len(lut)} unique RGB keys to {output_json_path}")

if __name__ == "__main__":
    main()   