import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
import utils
import pandas as pd
import cv2
from PIL import Image
import os

# Page Config
st.set_page_config(
    page_title="SR-IQA Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 SR-IQA Visualizer")
st.markdown(
    "图像超分质量评价工作台：上传 GT 和 SR 图像，一键计算 PSNR/SSIM/LPIPS 并进行可视化对比。"
)

# --- Sidebar ---
with st.sidebar:
    st.header("1. 设置 (Settings)")

    # Metrics Selection
    available_metrics = [
        "PSNR",
        "SSIM",
        "LPIPS",
        "FID",
        "CLIPIQA",
        "CNNIQA",
        "MUSIQ",
        "DISTS",
    ]
    selected_metrics = st.multiselect(
        "Select Metrics", available_metrics, default=["PSNR", "SSIM", "LPIPS"]
    )

    lpips_net = st.selectbox("LPIPS Backbone", ["vgg", "alex"], index=0)
    crop_border = st.number_input(
        "Crop Border (px)", min_value=0, value=4, help="SR 常用评估设置，切除边缘像素"
    )
    use_y_channel = st.checkbox(
        "Convert to Y-channel for PSNR/SSIM",
        value=True,
        help="SR 论文通常在 Y 通道计算指标",
    )

    st.header("2. 图像输入 (Input)")
    input_mode = st.radio(
        "Input Mode", ["Upload File", "Server Path", "Server Folder"], index=0
    )

    gt_file = None
    sr_file = None
    gt_path = None
    sr_path = None

    # Initialize session state for folder navigation
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if input_mode == "Upload File":
        gt_file = st.file_uploader(
            "Upload Reference (GT)", type=["png", "jpg", "jpeg", "bmp", "tiff"]
        )
        sr_file = st.file_uploader(
            "Upload Distorted (SR)", type=["png", "jpg", "jpeg", "bmp", "tiff"]
        )
    elif input_mode == "Server Path":
        gt_path = st.text_input("GT Image Path", placeholder="/path/to/gt.png")
        sr_path = st.text_input("SR Image Path", placeholder="/path/to/sr.png")
    elif input_mode == "Server Folder":
        gt_folder = st.text_input("GT Folder Path", placeholder="/path/to/gt_folder")
        sr_folder = st.text_input("SR Folder Path", placeholder="/path/to/sr_folder")

        if gt_folder and sr_folder:
            if os.path.isdir(gt_folder) and os.path.isdir(sr_folder):
                # Get file lists
                gt_files = utils.get_image_files(gt_folder)

                # Filter files that exist in both folders (assuming same filename)
                valid_files = [
                    f for f in gt_files if os.path.exists(os.path.join(sr_folder, f))
                ]

                if not valid_files:
                    st.error(
                        "No matching image files found in both folders (filenames must match)."
                    )
                else:
                    st.sidebar.markdown(f"**Found {len(valid_files)} matching images**")

                    # Ensure index is valid
                    if st.session_state.current_index >= len(valid_files):
                        st.session_state.current_index = 0

                    # Navigation Buttons
                    col_prev, col_next = st.sidebar.columns(2)
                    if col_prev.button("⬅️ Previous"):
                        st.session_state.current_index = max(
                            0, st.session_state.current_index - 1
                        )
                    if col_next.button("Next ➡️"):
                        st.session_state.current_index = min(
                            len(valid_files) - 1, st.session_state.current_index + 1
                        )

                    # Display current file info
                    current_file = valid_files[st.session_state.current_index]
                    st.sidebar.info(
                        f"Current: `{current_file}`\n({st.session_state.current_index + 1}/{len(valid_files)})"
                    )

                    # Set paths for loading
                    gt_path = os.path.join(gt_folder, current_file)
                    sr_path = os.path.join(sr_folder, current_file)
            else:
                st.error("Invalid folder path(s).")

# --- Main Content ---
img_gt_raw = None
img_sr_raw = None

try:
    if input_mode == "Upload File":
        if gt_file and sr_file:
            img_gt_raw = utils.load_image(gt_file)
            img_sr_raw = utils.load_image(sr_file)
    elif input_mode == "Server Path" or input_mode == "Server Folder":
        if gt_path and sr_path:
            img_gt_raw = utils.load_image_from_path(gt_path)
            img_sr_raw = utils.load_image_from_path(sr_path)
except Exception as e:
    st.error(f"Error loading images: {e}")

if img_gt_raw is not None and img_sr_raw is not None:
    # Preprocess (Resize & Crop)
    img_gt, img_sr = utils.preprocess_images(
        img_gt_raw, img_sr_raw, crop_border=crop_border
    )

    # --- Metrics Dashboard ---
    st.subheader("📊 全局指标 (Metrics)")

    with st.spinner("正在计算指标..."):
        metrics = utils.calculate_metrics(
            img_gt,
            img_sr,
            use_y_channel=use_y_channel,
            lpips_net=lpips_net,
            selected_metrics=selected_metrics,
        )

    # Dynamic display of metrics
    if metrics:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            # Determine delta color (LPIPS, FID, DISTS are lower is better)
            lower_is_better = name in ["LPIPS", "FID", "DISTS"]
            delta_color = "inverse" if lower_is_better else "normal"

            # Format value
            if np.isnan(value):
                display_val = "N/A"
            else:
                display_val = f"{value:.4f}"
                if name == "PSNR":
                    display_val += " dB"

            col.metric(name, display_val, delta_color=delta_color)
    else:
        st.warning("No metrics selected.")

    # Export Data
    metrics_df = pd.DataFrame([metrics])
    st.download_button(
        label="Download Metrics as CSV",
        data=metrics_df.to_csv(index=False),
        file_name="metrics.csv",
        mime="text/csv",
    )

    # --- Visual Comparison ---
    st.subheader("👁️ 可视化对比 (Visual Comparison)")

    tab1, tab2, tab3 = st.tabs(
        ["↔️ Slider Comparison", "🔥 Error Heatmap", "📈 FFT Spectrum"]
    )

    with tab1:
        st.write("左右拖动滑块对比细节：")
        # streamlit-image-comparison expects images in RGB
        image_comparison(
            img1=img_gt,
            img2=img_sr,
            label1="Reference (GT)",
            label2="Distorted (SR)",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )

    with tab2:
        st.write("差值热力图 (Absolute Difference): 颜色越亮表示误差越大。")
        error_map = utils.get_error_map(img_gt, img_sr)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(error_map, cmap="jet")
        plt.colorbar(im, ax=ax)
        ax.axis("off")
        st.pyplot(fig)

    with tab3:
        st.write("频域分析 (FFT): 检查高频信息丢失或伪影。")
        fft_gt = utils.get_fft_spectrum(img_gt)
        fft_sr = utils.get_fft_spectrum(img_sr)

        col_fft1, col_fft2 = st.columns(2)

        with col_fft1:
            st.caption("GT Spectrum")
            fig1, ax1 = plt.subplots()
            ax1.imshow(fft_gt, cmap="gray")
            ax1.axis("off")
            st.pyplot(fig1)

        with col_fft2:
            st.caption("SR Spectrum")
            fig2, ax2 = plt.subplots()
            ax2.imshow(fft_sr, cmap="gray")
            ax2.axis("off")
            st.pyplot(fig2)

else:
    st.info("👈 请在左侧侧边栏上传 GT 和 SR 图像以开始分析。")
