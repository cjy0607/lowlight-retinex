"""快速引导滤波（O(1)/像素，基于盒式滤波）。"""
import cv2
import numpy as np


def guided_filter(I, p, r, eps):
    """单通道引导滤波：输出 q ≈ a*I + b，保持 I 的边缘。"""
    I = np.ascontiguousarray(I, dtype=np.float32)
    p = np.ascontiguousarray(p, dtype=np.float32)
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    varI = cv2.boxFilter(I * I, -1, k) - mI * mI
    covIp = cv2.boxFilter(I * p, -1, k) - mI * mp
    a = covIp / (varI + eps)
    b = mp - a * mI
    ma = cv2.boxFilter(a, -1, k)
    mb = cv2.boxFilter(b, -1, k)
    return ma * I + mb


def guided_filter_rgb(guide, src, r, eps):
    """彩色引导滤波（逐通道），guide/src 均为 float32 [0,1]。"""
    out = np.empty_like(src, dtype=np.float32)
    for c in range(src.shape[2]):
        out[..., c] = guided_filter(guide[..., c], src[..., c], r, eps)
    return out
