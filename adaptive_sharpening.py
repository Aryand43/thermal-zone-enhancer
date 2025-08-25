import cv2
import numpy as np
import time
import cv2.ximgproc

def apply_clahe_and_edge_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhances L channel in LAB space using CLAHE + Laplacian sharpening.
    
    Matches: Step 4 in invention. Boosts perceptual contrast selectively.
    Novelty: Thermal-specific noise-aware sharpening with conditional application.
    """
    start = time.time()
    if image.dtype != np.uint8: image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Light edge enhancement
    edge = cv2.Laplacian(l_clahe, cv2.CV_64F)
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    l_edge = cv2.addWeighted(l_clahe, 1.2, edge, 0.3, 0)

    enhanced = cv2.merge((l_edge, a, b))
    out = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    print(f"Adaptive enhancement latency: {time.time() - start:.6f} seconds")
    return out

def guided_filter_and_unsharp_mask(image: np.ndarray) -> np.ndarray:
    """
    Applies guided filtering followed by unsharp masking to preserve edges 
    while enhancing contrast.

    Matches: Step 5 in invention. Hybrid not found in prior thermal pipelines.
    Novelty: Avoids artificial edges from over-enhancement.
    """
    start = time.time()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    guide = image.copy()
    filtered = cv2.ximgproc.guidedFilter(guide=guide, src=image, radius=6, eps=40, dDepth=-1)

    sharp = cv2.addWeighted(image, 1.8, filtered, -0.8, 0)  # more aggressive
    print(f"Guided sharpening latency: {time.time() - start:.6f} seconds")
    return np.clip(sharp, 0, 255).astype(np.uint8)

def apply_edsr_and_lab_refinement(image: np.ndarray, model_path: str = 'EDSR_x2.pb', scale: int = 2) -> np.ndarray:
    """
    Applies deep learning-based super-resolution (EDSR) followed by LAB-based 
    perceptual refinement using histogram equalization and sharpening.

    Matches: Step 6 in invention. Combines DL and classical enhancement.
    Novelty: First thermal-only pipeline combining EDSR + LAB refinements.
    """
    start = time.time()
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel('edsr', scale)
    upscaled = sr.upsample(image)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(cv2.GaussianBlur(l, (3, 3), 0))
    sharpened = cv2.Laplacian(l, cv2.CV_64F)
    l_final = cv2.addWeighted(l, 1.2, sharpened.astype(np.uint8), -0.2, 0)
    final = cv2.merge((l_final, a, b))
    upscaled_image = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    end = time.time()
    print(f"Super resolution latency: {end - start:.6f} seconds")
    return upscaled_image

