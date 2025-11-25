
# <img align="left" width="135" height="120" src="fig/logo2.png"> IQA-Visualization

A lightweight web-based tool for Image Quality Assessment (IQA)

<a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit"></a>
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
<a href="https://github.com/chaofengc/IQA-PyTorch"><img src="https://img.shields.io/badge/Backend-PyIQA-blue?style=flat&logo=github&logoColor=white" alt="IQA-PyTorch"></a>
<a href="https://github.com/XuBao12/IQA-Visualization"><img src="https://img.shields.io/github/stars/XuBao12/IQA-Visualization?style=social" alt="GitHub Stars"></a>


## 📖 Introduction

**SR-IQA Visualization** is built with Streamlit and PyIQA, providing a comprehensive suite of metrics and visualization tools for researchers to evaluate and compare model results efficiently.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="fig/app1.png" alt="App Screenshot 1" width="100%"/>
        <br>
      </td>
      <td align="center">
        <img src="fig/app2.png" alt="App Screenshot 2" width="100%"/>
        <br>
      </td>
    </tr>
  </table>
</div>

## ✨ Features

### 1. Comprehensive Metrics Support

Calculate a wide range of Full-Reference (FR) and No-Reference (NR) metrics:

- **Standard**: PSNR, SSIM
- **Perceptual**: LPIPS, DISTS, FID
- **No-Reference**: CLIPIQA, CNNIQA, MUSIQ
- _Customizable selection via sidebar._

### 2. Flexible Input Modes

- **Upload File**: Drag & drop local images.
- **Server Path**: Direct input of absolute file paths on the server.
- **Server Folder (Batch Mode)**: Point to GT and SR folders to browse images. Supports **Batch Evaluation** to calculate average metrics (including FID) for the entire dataset with one click.

### 3. Advanced Visualization

- **Slider Comparison**: Interactive "Before/After" slider to inspect restoration details.
- **Error Heatmap**: Visualize pixel-wise absolute differences.
- **FFT Spectrum**: Frequency domain analysis to detect artifacts or high-frequency loss.
- **ROI Crop & Zoom**: Draw a custom rectangular box on the reference image to inspect specific regions with 4x magnification.

## 🛠️ Installation

1.  **Clone the repository**:

    ```bash
    git clone git@github.com:XuBao12/IQA-Visualization.git
    cd IQA-Visualization
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Start the application**:

    ```bash
    python -m streamlit run app.py
    ```

2.  **Configure Settings (Sidebar)**:

    - **Select Metrics**: Choose which metrics to calculate (e.g., PSNR, SSIM, LPIPS, MUSIQ...).
    - **LPIPS Backbone**: VGG or Alex.
    - **Crop Border**: Pixels to ignore around the edges (standard practice in SR).

3.  **Select Input Mode**:
    - **Upload File**: Good for quick checks of local files.
    - **Server Path**: Good for checking specific files on the remote server.
    - **Server Folder**: Best for browsing results. Ensure filenames in GT and SR folders match.

## 📂 Project Structure

```
.
├── app.py              # Main Streamlit application
├── utils.py            # Helper functions (Metrics, I/O, Preprocessing)
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## 📝 Notes

- **First Run**: When selecting metrics like MUSIQ or FID for the first time, `pyiqa` will automatically download the pretrained weights. This may take a few minutes.
- **FID**: When using "Server Folder" mode, FID is calculated globally on the entire dataset distribution (standard practice). For single images, it is not computed.
- **Metric Standards**: PSNR/SSIM are calculated on the Y-channel (YCbCr) with border cropping by default, following standard SR academic protocols.
