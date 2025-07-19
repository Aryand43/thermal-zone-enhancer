# thermal-zone-enhancer

A thermal image processing pipeline designed for precision enhancement of regions of interest in `.tiff` images, with applications in materials science and melt pool analysis.

## Features

- Thermal zone detection using contour-based segmentation
- High-quality upscaling using `cv2.INTER_LANCZOS4`
- Perceptual enhancement in LAB color space
- Denoising using `cv2.fastNlMeansDenoisingColored`
- Optional sharpening applied to the lightness channel
- Batch processing with checkpointed output stages

## Pipeline Overview

1. **Detection and Cropping**  
   Extracts the largest thermal blob using contour area from grayscale-converted input.

2. **Quality Enhancement**  
   - Resizes using Lanczos4 interpolation  
   - Converts to LAB color space  
   - Applies denoising and sharpening on L channel  
   - Converts back to RGB

3. **Output**  
   - Cropped thermal zones saved to `checkpoint_1_detect_and_crop/`  
   - Enhanced results saved to `checkpoint_2_quality_enhancement/`

## Requirements

- numpy  
- opencv-python  
- time  
- (Optional) flirimageextractor if working with FLIR JPEG metadata

## Usage

Place `.tiff` images in the `data/` directory and run the main script.  
Enhanced outputs will be saved to designated checkpoint folders.