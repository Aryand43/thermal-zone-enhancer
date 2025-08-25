# Thermal Zone Enhancer

A modular pipeline for precision thermal image enhancement and temperature mapping. Built for thermal-only datasets (e.g. melt pool monitoring, additive manufacturing).

## What This Does

```
.tiff → [detect ROI] → [adaptive enhance] → [super-res] → [RGB-to-°C] → [track gradients]
```

1. Detects hottest zone in thermal image
2. Enhances resolution using CLAHE + Laplacian + EDSR
3. Converts RGB to °C via ΔE-based LUT mapping
4. Tracks hot pixel motion and gradient over time

## Folder Structure

```
input_thermal_images/             Raw .tiff or thermal .png
checkpoint_1_roi_crop/            ROI cropped (thermal zone only)
checkpoint_2_adaptive_enhance/    Enhanced (CLAHE + sharpening)
checkpoint_3_adaptive_postprocess/ Final EDSR + LAB-tuned output
temperature_matrices/             Per-pixel °C matrices (.npy)
lut_calibration/                  RGB to °C LUT (JSON)
thermal_gradient_outputs/         Velocity and gradient plots
```

## How to Run

```bash
python main.py
```

Runs full preprocessing pipeline:

* Detects thermal ROI
* Crops, enhances, upscales
* Saves intermediate outputs

```bash
python main_generate_temp_matrices.py
```

Uses RGB-to-temp LUT (`color_temp_lut.json`) to produce:

* `.npy` temperature matrix
* Optional `.png` heatmap

```bash
python main_track_thermal_gradient.py
```

Loads `.npy` matrices, tracks hot zones, and plots gradient/velocity trends.

## RGB to °C Mapping (ΔE Matching)

Uses LAB-space ΔE (CIEDE2000) to match each RGB pixel to the closest LUT entry. If ΔE < 25, assigns the corresponding °C from LUT.

**Note:** Accuracy depends heavily on LUT coverage. Generate it via `color_temp_mapping.py` by manually clicking on reference images to map RGB values to known temperatures.

## Key Modules

| File                              | Purpose                              |
| --------------------------------- | ------------------------------------ |
| `main.py`                         | Full pipeline runner                 |
| `main_generate_temp_matrices.py`  | RGB to °C matrix generator           |
| `main_track_thermal_gradient.py`  | Thermal motion analysis              |
| `color_temp_mapping.py`           | Manual LUT builder (click RGB ↔ °C)  |
| `temperature_matrix_generator.py` | ΔE-matching logic                    |
| `thermal_gradient_analysis.py`    | Hot pixel tracker                    |
| `adaptive_sharpening.py`          | CLAHE + Laplacian + guided filtering |
| `resolution_enhancement.py`       | LAB denoise + EDSR                   |
| `thermal_roi_detection.py`        | ROI crop using contours              |

## Limitations

* LUT-based mapping can miss pixels if ΔE exceeds threshold
* Motion tracker is currently 1D (horizontal shifts only)
* Super-resolution assumes availability of `.pb` EDSR checkpoint

## License

MIT

## Authors

Aryan Dutt, Shubham Chandra, Paulo Jorge Da Silva Bartolo
(Part of formal invention disclosure on adaptive thermal imaging pipelines)
