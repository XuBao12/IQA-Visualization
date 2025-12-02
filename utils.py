import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyiqa

# Global cache for metrics to avoid reloading
_metrics_cache = {}


def get_pyiqa_metric(metric_name, device=None, **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a unique key for the cache
    key = (metric_name, device, frozenset(kwargs.items()))

    if key not in _metrics_cache:
        _metrics_cache[key] = pyiqa.create_metric(metric_name, device=device, **kwargs)

    return _metrics_cache[key]


import os


def load_image(image_file):
    """Load image from UploadedFile and convert to RGB numpy array."""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_from_path(image_path):
    """Load image from server path and convert to RGB numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_images(img_gt, img_sr, crop_border=0):
    """
    Preprocess images:
    1. Resize SR to match GT if dimensions differ.
    2. Crop borders if requested.
    Returns processed numpy arrays (H, W, 3).
    """
    h_gt, w_gt = img_gt.shape[:2]
    h_sr, w_sr = img_sr.shape[:2]

    if (h_gt != h_sr) or (w_gt != w_sr):
        # Resize SR to match GT
        img_sr = cv2.resize(img_sr, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
        # img_gt = cv2.resize(img_gt, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

    if crop_border > 0:
        img_gt = img_gt[crop_border:-crop_border, crop_border:-crop_border, :]
        img_sr = img_sr[crop_border:-crop_border, crop_border:-crop_border, :]

    return img_gt, img_sr


def to_tensor(img):
    """Convert numpy image (H, W, C) 0-255 to tensor (1, C, H, W) 0-1."""
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def calculate_metrics(
    img_gt,
    img_sr,
    crop_border=0,
    use_y_channel=True,
    lpips_net="vgg",
    selected_metrics=None,
):
    """
    Calculate selected metrics using pyiqa.
    img_gt, img_sr: Numpy arrays (H, W, 3), RGB, 0-255.
    """
    if selected_metrics is None:
        selected_metrics = ["PSNR", "SSIM", "LPIPS"]

    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors (1, C, H, W) in [0, 1]
    tensor_gt = to_tensor(img_gt).to(device)
    tensor_sr = to_tensor(img_sr).to(device)

    try:
        # Define metric configurations
        # Key: Display Name, Value: (pyiqa_name, is_fr, kwargs)
        metric_configs = {
            "PSNR": ("psnr", True, {"test_y_channel": use_y_channel}),
            "SSIM": ("ssim", True, {"test_y_channel": use_y_channel}),
            "LPIPS": ("lpips", True, {"net": lpips_net}),
            "DISTS": ("dists", True, {}),
            "FID": ("fid", True, {}),  # Note: FID on single image pair is non-standard
            "CLIPIQA": ("clipiqa", False, {}),
            "CNNIQA": ("cnniqa", False, {}),
            "MUSIQ": ("musiq", False, {}),
        }

        for name in selected_metrics:
            if name == "SRE":
                try:
                    score = calculate_SRE(img_gt, img_sr)
                    metrics[name] = score
                except Exception as e:
                    print(f"Error calculating SRE: {e}")
                    metrics[name] = float("nan")

            if name not in metric_configs:
                continue

            pyiqa_name, is_fr, kwargs = metric_configs[name]

            try:
                metric_func = get_pyiqa_metric(pyiqa_name, device=device, **kwargs)

                with torch.no_grad():
                    if is_fr:
                        score = metric_func(tensor_sr, tensor_gt).item()
                    else:
                        score = metric_func(tensor_sr).item()

                metrics[name] = score
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                metrics[name] = float("nan")
    finally:
        # Cleanup tensors to free GPU memory
        del tensor_gt
        del tensor_sr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return metrics


def get_error_map(img_gt, img_sr):
    """Calculate absolute difference map."""
    # Convert to float
    diff = np.abs(img_gt.astype(np.float32) - img_sr.astype(np.float32))
    # Mean over channels to get intensity difference
    diff = np.mean(diff, axis=2)
    # Normalize for visualization (optional, but heatmap usually handles range)
    return diff


def get_fft_spectrum(img):
    """Compute Log-Magnitude Spectrum."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum


def get_1d_power_spectrum(img):
    """Compute 1D radially averaged power spectrum."""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    # Power spectrum
    psd2D = (np.abs(fshift) ** 2) / (img.shape[0] * img.shape[1])

    # Calculate radial profile
    h, w = psd2D.shape
    center_y, center_x = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(int)

    # Average PSD at each radius
    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())

    # Avoid division by zero
    radial_profile = tbin / (nr + 1e-8)

    return radial_profile


def get_image_files(folder_path):
    """Return a sorted list of image files in the folder."""
    if not os.path.exists(folder_path):
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    files.sort()
    return files


def calculate_fid_folder(gt_folder, sr_folder, device=None):
    """Calculate FID score between two folders."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Create FID metric
        fid_metric = pyiqa.create_metric("fid", device=device)

        # Calculate FID (distorted_path, reference_path)
        with torch.no_grad():
            score = fid_metric(sr_folder, gt_folder)

        return score.item() if isinstance(score, torch.Tensor) else score
    except Exception as e:
        print(f"Error calculating FID for folders: {e}")
        return None


def get_texture_analysis(img):
    """
    Analyze texture using Gabor filters.
    Returns:
        energy_vis: RGB image of texture energy (Heatmap).
        orientation_vis: RGB image of dominant orientation (Color-coded).
    """
    # Define filters
    filters = []
    ksize = 21  # Kernel size
    num_thetas = 8  # Number of orientations
    for i in range(num_thetas):
        theta = i * np.pi / num_thetas
        # sigma, theta, lambd, gamma, psi
        kern = cv2.getGaborKernel(
            (ksize, ksize), 3.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F
        )
        filters.append(kern)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    accum_energy = np.zeros_like(gray, dtype=np.float32)
    max_response = np.zeros_like(gray, dtype=np.float32) - 1e9
    orientation_idx = np.zeros_like(gray, dtype=np.uint8)

    for i, kern in enumerate(filters):
        fimg = cv2.filter2D(gray, cv2.CV_32F, kern)

        # Energy (Sum of squared responses)
        accum_energy += fimg * fimg

        # Orientation (Index of max response)
        mask = fimg > max_response
        max_response[mask] = fimg[mask]
        orientation_idx[mask] = i

    # Visualize Energy
    # Normalize to 0-255
    energy_norm = cv2.normalize(accum_energy, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    energy_vis = cv2.applyColorMap(energy_norm, cv2.COLORMAP_INFERNO)
    energy_vis = cv2.cvtColor(energy_vis, cv2.COLOR_BGR2RGB)

    # Visualize Orientation
    # Use HSV: Hue = Orientation, Value = Response Strength
    hue = (orientation_idx.astype(np.float32) / num_thetas * 180).astype(np.uint8)
    sat = np.ones_like(hue) * 255
    # Normalize max response for Value channel
    val = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv = cv2.merge([hue, sat, val])
    orientation_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return energy_vis, orientation_vis


def get_edge_analysis(img, method="Canny"):
    """
    Extract edge map using specified method.
    method: 'Canny', 'Sobel', 'Laplacian'
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if method == "Canny":
        # Use median-based automatic thresholding
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        return edges
    elif method == "Sobel":
        # Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        # Normalize to 0-255
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.abs(lap)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return lap
    else:
        return gray


def calculate_SRE(img_gt, img_sr):
    """
    计算 GT 和 SR 图像在频域上的 L1 距离 (Log-PSD MAE)。

    Args:
        img_gt (np.ndarray): Ground Truth 图片 (H, W, 3) 或 (H, W), 范围 0-255.
        img_sr (np.ndarray): 生成的图片 (H, W, 3) 或 (H, W), 范围 0-255.

    Returns:
        float: 频域距离分数 (越低越好).
        (np.array, np.array): 返回 GT 和 SR 的 1D 曲线数据，方便画图分析.
    """
    psd_gt = get_1d_power_spectrum(img_gt)
    psd_sr = get_1d_power_spectrum(img_sr)

    min_len = min(len(psd_gt), len(psd_sr))
    curve_gt = psd_gt[1:min_len]
    curve_sr = psd_sr[1:min_len]
    curve_gt = 10 * np.log10(psd_gt + 1e-10)
    curve_sr = 10 * np.log10(psd_sr + 1e-10)

    # 计算平均绝对误差
    score = np.mean(np.abs(curve_gt - curve_sr))

    return score

def calculate_gsd(gt_paths, sr_paths, target_size=(512, 512)):
    """
    计算 GSD (Global Spectral Distance) - 数据集层面的全局频域距离。

    逻辑：
    1. 统一 Resize 所有图片到 target_size (对齐频域刻度)。
    2. 计算每张图的 原始功率 (Raw Power, |F|^2)。
    3. 累加所有图的 1D 功率曲线。
    4. 求平均，得到 "数据集平均功率谱"。
    5. 转 dB。
    6. 计算 L1 距离。

    Args:
        gt_paths (list): GT 图片的路径列表。
        gen_paths (list): 生成图片的路径列表。
        target_size (tuple): 统一的计算尺寸，默认 (512, 512)。
                             建议设为数据集中大部分图片的最小尺寸或标准尺寸。

    Returns:
        float: GSD 分数 (dB)。
        tuple: (gt_curve_db, gen_curve_db) 用于画图。
    """

    # --- 内部辅助函数 1: 预处理 ---
    def preprocess_image(path):
        # 读取图片
        img = cv2.imread(path)
        if img is None:
            return None

        # 1. 转灰度 (只分析亮度信息的纹理)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 归一化 (0~1)
        img = img.astype(np.float32) / 255.0

        # 3. 统一 Resize (关键步骤！)
        # 使用 INTER_AREA，因为它在调整大小时对频域信息的破坏最小
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        return img
    # ================= 核心流程 =================

    # 1. 处理 GT 数据集
    print(f"Calculating GSD: Processing {len(gt_paths)} GT images...")
    sum_profile_gt = None
    count_gt = 0

    for path in gt_paths:
        img = preprocess_image(path)
        if img is None: continue

        profile = get_1d_power_spectrum(img)

        if sum_profile_gt is None:
            sum_profile_gt = np.zeros_like(profile)

        # 累加长度可能不一致（虽然resize了，但bincount结果取决于对角线长度）
        # 加上 padding 保护
        if len(profile) > len(sum_profile_gt):
            profile = profile[:len(sum_profile_gt)]
        elif len(profile) < len(sum_profile_gt):
            sum_profile_gt = sum_profile_gt[:len(profile)]

        sum_profile_gt += profile
        count_gt += 1

    avg_profile_gt = sum_profile_gt / count_gt

    # 2. 处理 Gen 数据集
    print(f"Calculating GSD: Processing {len(sr_paths)} Generated images...")
    sum_profile_sr = None
    count_sr = 0

    for path in sr_paths:
        img = preprocess_image(path)
        if img is None: continue

        profile = get_1d_power_spectrum(img)

        if sum_profile_sr is None:
            sum_profile_sr = np.zeros_like(profile)

        # 长度对齐保护
        length = min(len(profile), len(sum_profile_sr))
        sum_profile_sr = sum_profile_sr[:length] + profile[:length]
        count_sr += 1

    avg_profile_sr = sum_profile_sr / count_sr

    # ================= 计算指标 =================

    # 3. 截断无效区域
    # 因为 Resize 到了 target_size，对角线长度是固定的
    # 通常取半径的前 85% 比较稳妥，去掉边角的高频伪影
    min_len = min(len(avg_profile_gt), len(avg_profile_sr))
    valid_len = int(min_len * 0.85)

    # 4. 转 dB (先平均，后 Log) - 这是 GSD 的精髓
    # 跳过 index 0 (DC分量/平均亮度)
    curve_gt_db = 10 * np.log10(avg_profile_gt[1:valid_len] + 1e-10)
    curve_sr_db = 10 * np.log10(avg_profile_sr[1:valid_len] + 1e-10)

    # 5. 计算 L1 距离 (MAE)
    gsd_score = np.mean(np.abs(curve_gt_db - curve_sr_db))

    return gsd_score
