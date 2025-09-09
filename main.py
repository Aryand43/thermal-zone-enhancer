import os
import cv2
import numpy as np
from typing import Tuple

from thermal_roi_detection import detect_thermal_zone, crop_to_zone
from resolution_enhancement import upsample_using_lanczos
from adaptive_sharpening import (
    apply_clahe_and_edge_enhancement,
    guided_filter_and_unsharp_mask,
    apply_edsr_and_lab_refinement
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
            # Resolution enhancement
            enhanced = upsample_using_lanczos(image, scale=4)
            output_path_stage2 = os.path.join(output2_dir, filename)
            cv2.imwrite(output_path_stage2, enhanced)

            # Adaptive enhancement and post-processing
            adaptively_enhanced = apply_clahe_and_edge_enhancement(enhanced)
            sharpened = guided_filter_and_unsharp_mask(adaptively_enhanced)
            final_output = apply_edsr_and_lab_refinement(sharpened)

            output_path_stage3 = os.path.join(output3_dir, filename)
            cv2.imwrite(output_path_stage3, final_output)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    run_detection_and_crop()
    run_quality_enhancement_and_final_postprocess()
