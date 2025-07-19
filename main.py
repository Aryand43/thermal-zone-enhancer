import os
import cv2
import numpy as np
from typing import Tuple
from checkpoint1 import detect_thermal_zone, crop_to_zone
from checkpoint2 import enhance_resolution

input_dir = "data"
output1_dir = "checkpoint_1_detect_and_crop"
output2_dir = "checkpoint_2_quality_enhancement"

os.makedirs(output1_dir, exist_ok=True)
os.makedirs(output2_dir, exist_ok=True)

def run_detection_and_crop():
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        try:
            bbox = detect_thermal_zone(image)
            cropped = crop_to_zone(image, bbox)
            output_path = os.path.join(output1_dir, filename)
            cv2.imwrite(output_path, cropped)
        except:
            continue

def run_quality_enhancement():
    enhance_files = [f for f in os.listdir(output1_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    for filename in enhance_files:
        input_path = os.path.join(output1_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        try:
            enhanced = enhance_resolution(image, scale=4)
            output_path = os.path.join(output2_dir, filename)
            cv2.imwrite(output_path, enhanced)
        except:
            continue

if __name__ == "__main__":
    run_detection_and_crop()
    run_quality_enhancement()