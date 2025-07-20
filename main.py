import os
import cv2
import numpy as np
from typing import Tuple
from checkpoint1 import detect_thermal_zone, crop_to_zone
from checkpoint2 import enhance_resolution
from checkpoint3 import adaptive_enhance_thermal_image, guided_sharpen_thermal_image

input_dir = "data"
output1_dir = "checkpoint_1_detect_and_crop"
output2_dir = "checkpoint_2_quality_enhancement"
output3_dir = "checkpoint_3_adaptive_postprocess"

os.makedirs(output1_dir, exist_ok=True)
os.makedirs(output2_dir, exist_ok=True)
os.makedirs(output3_dir, exist_ok=True)

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

def run_quality_enhancement_and_final_postprocess():
    files = [f for f in os.listdir(output1_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    for filename in files:
        input_path = os.path.join(output1_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        try:
            enhanced = enhance_resolution(image, scale=4)
            adaptively_enhanced = adaptive_enhance_thermal_image(enhanced)
            final_image = guided_sharpen_thermal_image(adaptively_enhanced)
            output_path = os.path.join(output3_dir, filename)
            cv2.imwrite(output_path, final_image)
        except:
            continue

if __name__ == "__main__":
    run_detection_and_crop()
    run_quality_enhancement_and_final_postprocess()