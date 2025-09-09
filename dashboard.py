import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Config
TEMP_DIR = "temperature_matrices"
GRADIENT_DIR = "thermal_gradient_outputs"

st.set_page_config(layout="wide", page_title="Thermal Zone Dashboard")

st.title("Melt Pool Thermal Dashboard")

# Sidebar controls
frame_id = st.sidebar.number_input("Frame ID", min_value=1, max_value=89, value=1)
show_velocity = st.sidebar.checkbox("Show Velocity Graphs", value=True)
show_gradients = st.sidebar.checkbox("Show Thermal Gradients", value=True)

# Load temp matrix
matrix_path = os.path.join(TEMP_DIR, f"{frame_id}.npy")
if os.path.exists(matrix_path):
    temp_matrix = np.load(matrix_path)

    fig, ax = plt.subplots()
    im = ax.imshow(temp_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_title(f"Temperature Heatmap - Frame {frame_id}")
    st.pyplot(fig)
else:
    st.error(f"No matrix found for frame {frame_id}")

# Velocity plots
if show_velocity:
    v_unweighted = os.path.join(GRADIENT_DIR, "velocity_avg_unweighted.png")
    v_1385 = os.path.join(GRADIENT_DIR, "velocity_weighted_1385_1450.png")
    v_1600 = os.path.join(GRADIENT_DIR, "velocity_weighted_gt1600.png")
    cols = st.columns(3)
    for col, path, title in zip(cols, [v_unweighted, v_1385, v_1600],
                                ["Unweighted Velocity", "1385–1450°C Velocity", ">1600°C Velocity"]):
        if os.path.exists(path):
            col.image(path, caption=title, use_container_width=True)

# Gradient arrays
if show_gradients:
    grad_path = os.path.join(GRADIENT_DIR, "thermal_gradients.npy")
    if os.path.exists(grad_path):
        gradients = np.load(grad_path, allow_pickle=True)
        st.line_chart([np.mean(g) if len(g) > 0 else 0 for g in gradients])
    else:
        st.warning("No gradient data found. Run main_track_thermal_gradient.py first.")

st.subheader("Raw Velocity Data")
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