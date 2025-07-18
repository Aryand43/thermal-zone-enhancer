import os
import cv2
import numpy as np
from typing import Tuple
from image_processing import detect_thermal_zone, crop_to_zone

input_dir = "data"
output_dir = "checkpoint_1_detect_and_crop"
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
for filename in image_files:
    input_path = os.path.join(input_dir, filename)
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Skipping unreadable image: {filename}")
        continue

    try:
        bbox = detect_thermal_zone(image)
        cropped = crop_to_zone(image, bbox)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped)
        print(f"Saved cropped thermal zone: {output_path}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__=="__main__":
    data_dir = "data"
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]