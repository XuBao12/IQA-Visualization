# SR-IQA Visualizer (图像超分质量评价工作台)

A lightweight web tool for Image Super-Resolution (SR) and Restoration quality assessment. It allows researchers to upload "Reference (GT)" and "Distorted (SR)" images to calculate metrics and visualize differences interactively.

## Features

- **Metrics Calculation**: Automatically calculates PSNR (Y-channel/RGB), SSIM, and LPIPS (VGG/Alex).
- **Visual Comparison**: Interactive "Before/After" slider.
- **Error Analysis**: Heatmap visualization of pixel-wise differences.
- **Frequency Domain**: FFT Spectrum analysis to check for high-frequency artifacts.
- **ROI Zoom**: Inspect local details with digital zoom.

## Installation

1.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  Run the app:
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Open the app in your browser (usually `http://localhost:8501`).
2.  Use the **Sidebar** to configure settings (LPIPS backbone, Crop Border).
3.  Upload your **Reference (GT)** and **Distorted (SR)** images.
4.  View metrics and explore the visualizations in the main area.
