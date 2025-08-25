import os
import cv2
import numpy as np
from typing import Tuple
from thermal_roi_detection import detect_thermal_zone, crop_to_zone
from resolution_enhancement import enhance_resolution
from adaptive_sharpening import (
    adaptive_enhance_thermal_image,
    guided_sharpen_thermal_image,
    apply_super_resolution
)

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
            output_path_stage2 = os.path.join(output2_dir, filename)
            cv2.imwrite(output_path_stage2, enhanced)

            adaptively_enhanced = adaptive_enhance_thermal_image(enhanced)
            sharpened = guided_sharpen_thermal_image(adaptively_enhanced)
            super_res = apply_super_resolution(sharpened, model_path='EDSR_x4.pb', scale=4)

            output_path_stage3 = os.path.join(output3_dir, filename)
            cv2.imwrite(output_path_stage3, super_res)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    run_detection_and_crop()
    run_quality_enhancement_and_final_postprocess()