# thermal-zone-enhancer

A precision thermal image enhancement pipeline for extracting and improving regions of interest in `.tiff` images. Built for materials science and melt pool analysis.

## Features
- Thermal zone detection via contour-based segmentation
- High-quality upscaling with `cv2.INTER_LANCZOS4`
- LAB color space perceptual enhancement
- Adaptive denoising using `cv2.fastNlMeansDenoisingColored`
- Edge-aware filtering and unsharp masking for fine structure sharpening
- Modular batch processing with checkpointed outputs

## Pipeline Overview

### 1. Detection and Cropping
- Converts input to grayscale
- Extracts the largest thermal region using contour area
- Saves to `checkpoint_1_detect_and_crop/`

### 2. Quality Enhancement
- Upscales image via Lanczos4 interpolation
- Converts to LAB color space
- Applies adaptive denoising and CLAHE on L channel
- Sharpens low-contrast images selectively
- Saves to `checkpoint_2_quality_enhancement/`

### 3. Guided Post-Processing
- Applies guided filtering and unsharp masking for detail preservation
- Final results saved to `checkpoint_3_adaptive_postprocess/`

## Requirements
- numpy
- opencv-python
- opencv-contrib-python
- time
- (optional) flirimageextractor — for FLIR JPEG thermal metadata

## Usage
1. Place `.tiff` images in the `data/` directory.
2. Run `main.py`.
3. Enhanced outputs will be saved in corresponding checkpoint folders.