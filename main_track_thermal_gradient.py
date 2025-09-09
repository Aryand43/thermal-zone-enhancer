import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.ndimage import gaussian_filter
import argparse
import shutil

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--frame_rate", type=float, default=80.0)
parser.add_argument("--pixel_resolution_um", type=float, default=80.0)
parser.add_argument("--liquidus_temp", type=float, default=1800.0)
parser.add_argument("--solidus_temp", type=float, default=1450.0)
parser.add_argument("--threshold_mask", action="store_true")
parser.add_argument("--superres", action="store_true")
args = parser.parse_args()

# === Paths ===
TEMP_DIR = "temperature_matrices"
GRADIENT_DIR = "thermal_gradient_outputs"
FRAME_DIR = "video_frames"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(GRADIENT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# === Clean Old Files ===
for folder in [TEMP_DIR, GRADIENT_DIR, FRAME_DIR]:
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

def extract_frames_from_video(video_path: str, output_dir: str) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        path = os.path.join(output_dir, f"frame_{idx+1}.png")
        cv2.imwrite(path, gray)
        frame_paths.append(path)
        idx += 1
    cap.release()
    return frame_paths

def compute_temperature_matrix(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    frame = cv2.normalize(frame, None, 1000, args.liquidus_temp, cv2.NORM_MINMAX)
    if args.threshold_mask:
        frame[frame < args.solidus_temp] = 0
    return frame

def detect_thermal_zone(temperature_matrix: np.ndarray) -> Tuple[int, int, int, int]:
    blurred = cv2.GaussianBlur(temperature_matrix, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, args.solidus_temp, 2000, cv2.THRESH_BINARY)
    thresh_uint8 = np.uint8((thresh > 0) * 255)
    if thresh_uint8.ndim == 3:
        thresh_uint8 = cv2.cvtColor(thresh_uint8, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(thresh_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)
    return (0, 0, temperature_matrix.shape[1], temperature_matrix.shape[0])

def compute_velocity_and_gradients(
    temp_matrices: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]],
    dt: float,
    pixel_resolution_um: float
):
    velocity_unweighted, velocity_w1, velocity_w2, gradients = [], [], [], []
    for i in range(1, len(temp_matrices)):
        x, y, w, h = bboxes[i]
        prev = temp_matrices[i - 1][y:y+h, x:x+w]
        curr = temp_matrices[i][y:y+h, x:x+w]
        delta = curr - prev
        dx = delta / dt
        g = np.linalg.norm(dx, axis=0) if dx.ndim == 3 else dx
        mean_v = np.mean(g)
        velocity_unweighted.append(mean_v)
        mask1 = (curr >= 1385) & (curr <= 1450)
        mask2 = curr > 1600
        v1 = np.mean(g[mask1]) if mask1.shape == g.shape and np.any(mask1) else 0
        v2 = np.mean(g[mask2]) if mask2.shape == g.shape and np.any(mask2) else 0
        velocity_w1.append(v1)
        velocity_w2.append(v2)
        gradients.append(np.abs(curr - prev))
    return velocity_unweighted, velocity_w1, velocity_w2, gradients

def save_visualizations(v_u, v_w1, v_w2, gradients):
    np.save(os.path.join(GRADIENT_DIR, "velocity_unweighted.npy"), v_u)
    np.save(os.path.join(GRADIENT_DIR, "velocity_weighted_1385_1450.npy"), v_w1)
    np.save(os.path.join(GRADIENT_DIR, "velocity_weighted_gt1600.npy"), v_w2)
    np.save(os.path.join(GRADIENT_DIR, "thermal_gradients.npy"), gradients)
    for name, data in zip([
        "velocity_avg_unweighted.png",
        "velocity_weighted_1385_1450.png",
        "velocity_weighted_gt1600.png"], [v_u, v_w1, v_w2]):
        plt.figure()
        plt.plot(data)
        plt.xlabel("Frame")
        plt.ylabel("Velocity")
        plt.title(name.replace("_", " ").replace(".png", ""))
        plt.savefig(os.path.join(GRADIENT_DIR, name))
        plt.close()

def main():
    dt = 1.0 / args.frame_rate
    resolution = args.pixel_resolution_um
    frame_paths = extract_frames_from_video(args.video_path, FRAME_DIR)
    temp_matrices, bboxes = [], []
    for i, path in enumerate(frame_paths):
        from adaptive_sharpening import apply_clahe_and_edge_enhancement

        img_raw = cv2.imread(path)
        img = apply_clahe_and_edge_enhancement(img_raw)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        temp = compute_temperature_matrix(img)
        np.save(os.path.join(TEMP_DIR, f"{i+1:04d}.npy"), temp)
        temp_matrices.append(temp)
        bboxes.append(detect_thermal_zone(temp))
    v_u, v_w1, v_w2, gradients = compute_velocity_and_gradients(temp_matrices, bboxes, dt, resolution)
    save_visualizations(v_u, v_w1, v_w2, gradients)

if __name__ == "__main__":
    main()
