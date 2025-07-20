import cv2
import numpy as np
import time

def enhance_resolution(image: np.ndarray, scale: int) -> np.ndarray:
    start = time.time()
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input must be a non-empty NumPy array.")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] != 3:
        raise ValueError("Unsupported image format.")
    h, w = image.shape[:2]
    resized = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    denoised = cv2.fastNlMeansDenoisingColored(lab, None, 10, 10, 7, 21)
    l, a, b = cv2.split(denoised)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    l_sharp = cv2.filter2D(l, -1, kernel)
    merged = cv2.merge((l_sharp, a, b))
    enhanced_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    end = time.time()
    print(f"Enhancement latency: {end - start:.6f} seconds")
    return enhanced_image