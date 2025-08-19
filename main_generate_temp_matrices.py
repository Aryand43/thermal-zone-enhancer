import os
from generate_temperature_matrix import generate_temperature_matrix_from_roi

input_dir = "checkpoint_1_detect_and_crop"
output_dir = "temperature_matrices"
lut_path = "color_temp_lut.json"

os.makedirs(output_dir, exist_ok=True)

for i in range(1, 90): 
    image_path = f"{input_dir}/{i}.tiff"
    save_path = f"{output_dir}/{i}.npy"

    try:
        generate_temperature_matrix_from_roi(image_path, lut_path, save_path)
    except Exception as e:
        print(f"[ERROR] Failed on {i}.tiff: {e}")
