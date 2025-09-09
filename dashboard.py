import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import shutil
import subprocess
import cv2

st.set_page_config(layout="wide", page_title="Melt Pool Thermal Dashboard")
st.title("Melt Pool Thermal Dashboard")

# === Sidebar: Material and Calibration Parameters ===
st.sidebar.header("Material & Calibration Config")

liquidus_temperature = st.sidebar.number_input("Liquidus Temperature (°C)", value=1800, min_value=1000, max_value=3000)
solidus_temperature = st.sidebar.number_input("Solidus Temperature (°C)", value=1450, min_value=1000, max_value=3000)
pixel_resolution_um = st.sidebar.number_input("Pixel Resolution (µm)", value=80, min_value=1, max_value=500)
frame_rate = st.sidebar.number_input("Frame Rate (FPS)", value=80, min_value=1, max_value=1000)
threshold_temp_filtering = st.sidebar.checkbox("Apply Threshold Mask (<1450°C)", value=True)
superresolution_enabled = st.sidebar.checkbox("Enable Super-Resolution Filtering", value=False)

# === Upload Video ===
st.subheader("Upload Melt Pool Video")
video_file = st.file_uploader("Upload a single-track thermal video", type=["mp4", "avi", "mov"])

# === Run Analysis Button ===
run_analysis = st.button("Run Melt Pool Analysis")

TEMP_DIR = "temperature_matrices"
GRADIENT_DIR = "thermal_gradient_outputs"

if run_analysis:
    with st.spinner("Analyzing video... please wait"):

        if video_file:
            temp_video_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_video_dir, video_file.name)

            with open(video_path, "wb") as f:
                f.write(video_file.read())
        else:
            video_path = "test-vid.mp4"
            st.warning("No video uploaded, using fallback test-vid.mp4")

        # === Validate video ===
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count < 2:
            st.error(f"Video only has {frame_count} frame(s) — likely corrupted or invalid.")
        else:
            # Construct command with dynamic args
            cmd = [
                "python", "main_track_thermal_gradient.py",
                "--video_path", video_path,
                "--frame_rate", str(frame_rate),
                "--pixel_resolution_um", str(pixel_resolution_um),
                "--liquidus_temp", str(liquidus_temperature),
                "--solidus_temp", str(solidus_temperature)
            ]
            if threshold_temp_filtering:
                cmd.append("--threshold_mask")
            if superresolution_enabled:
                cmd.append("--superres")

            # Run and capture logs
            result = subprocess.run(cmd, capture_output=True, text=True)
            st.subheader("Backend Output:")
            st.code(result.stdout + "\n" + result.stderr)

            if video_file:
                shutil.rmtree(temp_video_dir)

    st.success("Analysis Complete")

# === Visualization Panel ===
if os.path.exists(TEMP_DIR):
    st.subheader("Frame-wise Temperature Visualization")
    frame_files = sorted([f for f in os.listdir(TEMP_DIR) if f.endswith(".npy")])
    
    if frame_files:
        frame_names = [f.replace(".npy", "") for f in frame_files]
        frame_id = st.selectbox("Select Frame ID", frame_names)
        temp_matrix = np.load(os.path.join(TEMP_DIR, f"{frame_id}.npy"))

        fig, ax = plt.subplots()
        im = ax.imshow(temp_matrix, cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Temperature (°C)")
        ax.set_title(f"Temperature Heatmap - Frame {frame_id}")
        st.pyplot(fig)

    # === Velocity Plots ===
    st.subheader("Velocity Maps")
    v_unweighted = os.path.join(GRADIENT_DIR, "velocity_avg_unweighted.png")
    v_1385 = os.path.join(GRADIENT_DIR, "velocity_weighted_1385_1450.png")
    v_1600 = os.path.join(GRADIENT_DIR, "velocity_weighted_gt1600.png")
    cols = st.columns(3)
    for col, path, title in zip(cols, [v_unweighted, v_1385, v_1600],
                                ["Unweighted", "1385–1450°C Weighted", ">1600°C Weighted"]):
        if os.path.exists(path):
            col.image(path, caption=title, use_container_width=True)

    # === Gradient Chart ===
    st.subheader("Thermal Gradient Trends")
    grad_path = os.path.join(GRADIENT_DIR, "thermal_gradients.npy")
    if os.path.exists(grad_path):
        gradients = np.load(grad_path, allow_pickle=True)
        st.line_chart([np.mean(g) if len(g) > 0 else 0 for g in gradients])

    # === Velocity Arrays ===
    st.subheader("Raw Velocity Arrays")
    col1, col2, col3 = st.columns(3)
    for col, file, label in zip(
        [col1, col2, col3],
        ["velocity_unweighted.npy", "velocity_weighted_1385_1450.npy", "velocity_weighted_gt1600.npy"],
        ["Unweighted", "1385–1450°C Weighted", ">1600°C Weighted"]
    ):
        full_path = os.path.join(GRADIENT_DIR, file)
        if os.path.exists(full_path):
            data = np.load(full_path, allow_pickle=True)
            col.line_chart([np.mean(v) if isinstance(v, (list, np.ndarray)) and len(v) > 0 else 0 for v in data])
            col.caption(label)

# === Solidification Boundary Velocities ===
st.subheader("Solidification Boundary Velocities")

vx_path = os.path.join(GRADIENT_DIR, "vx_solidification.npy")
vy_path = os.path.join(GRADIENT_DIR, "vy_solidification.npy")

if os.path.exists(vx_path) and os.path.exists(vy_path):
    vx = np.load(vx_path, allow_pickle=True)
    vy = np.load(vy_path, allow_pickle=True)

    mag_velocity = np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2)

    col1, col2, col3 = st.columns(3)
    col1.line_chart(vx)
    col1.caption("vx (µm/s)")

    col2.line_chart(vy)
    col2.caption("vy (µm/s)")

    col3.line_chart(mag_velocity)
    col3.caption("Velocity Magnitude (µm/s)")
else:
    st.info("Solidification velocity data not found. Ensure `solidification_tracker.py` has been executed.")
