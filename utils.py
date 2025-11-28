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
        # img_sr = cv2.resize(img_sr, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
        img_gt = cv2.resize(img_gt, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

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
