import cv2
import numpy as np
import time
import cv2.ximgproc

def adaptive_enhance_thermal_image(image: np.ndarray) -> np.ndarray:
    start = time.time()

    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Input must be a non-empty uint8 NumPy array.")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] != 3:
        raise ValueError("Unsupported image format.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    noise_std = np.std(gray)
    h = 4 if noise_std < 10 else 7 if noise_std < 20 else 10

    if noise_std >= 5:
        image = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=20, sigmaSpace=10)
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    l_clahe = clahe.apply(l)

    lap_var = cv2.Laplacian(l_clahe, cv2.CV_64F).var()
    if lap_var < 150:
        blurred = cv2.GaussianBlur(l_clahe, (3, 3), 0)
        l_sharp = cv2.addWeighted(l_clahe, 1.5, blurred, -0.5, 0)
    else:
        l_sharp = l_clahe

    final_lab = cv2.merge((l_sharp, a, b))
    final_image = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)

    end = time.time()
    print(f"Adaptive enhancement latency: {end - start:.6f} seconds")
    return final_image

def guided_sharpen_thermal_image(image: np.ndarray) -> np.ndarray:
    start = time.time()
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Input must be a non-empty uint8 NumPy array.")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] != 3:
        raise ValueError("Unsupported image format.")

    guide = image.copy()
    filtered = cv2.ximgproc.guidedFilter(guide=guide, src=image, radius=8, eps=100, dDepth=-1)

    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    blurred = cv2.GaussianBlur(l, (5, 5), sigmaX=1.0)
    l_sharp = cv2.addWeighted(l, 1.5, blurred, -0.5, 0)

    merged = cv2.merge((l_sharp, a, b))
    enhanced_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    end = time.time()
    print(f"Guided sharpening latency: {end - start:.6f} seconds")
    return enhanced_image