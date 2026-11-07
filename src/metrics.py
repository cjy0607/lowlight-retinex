"""客观评价指标：PSNR / SSIM / LOE / 信息熵 / 平均梯度 / 色彩丰富度。"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from .methods import luminance
except ImportError:
    from methods import luminance


def psnr(pred, gt):
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def ssim(pred, gt):
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))


def loe(pred, gt, k=50):
    """亮度序误差（Lightness Order Error），越小越好。"""
    p = cv2.resize(luminance(pred), (k, k), interpolation=cv2.INTER_AREA)
    g = cv2.resize(luminance(gt), (k, k), interpolation=cv2.INTER_AREA)
    dp = p[:, None] - p[None, :]
    dg = g[:, None] - g[None, :]
    mask = np.abs(dg) > 1e-4
    err = ((dp * dg) < 0).astype(np.float32)
    return float(err[mask].mean() * 100.0)


def entropy(pred):
    gray = np.clip(luminance(pred), 0, 1)
    hist, _ = np.histogram(gray, bins=256, range=(0, 1))
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def avg_gradient(pred):
    gray = np.clip(luminance(pred), 0, 1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0)))


def colorfulness(pred):
    r, g, b = pred[..., 0], pred[..., 1], pred[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def brightness(pred):
    return float(luminance(pred).mean())


METRIC_FUNCS = {
    "PSNR": psnr,
    "SSIM": ssim,
    "LOE": loe,
    "Entropy": entropy,
    "Gradient": avg_gradient,
    "Colorful": colorfulness,
    "Brightness": brightness,
}
