"""数据读取：LOLv1 配对图像。"""
import os

import cv2
import numpy as np


def to_float(img):
    return img.astype(np.float32) / 255.0


def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return to_float(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def list_pairs(root, split, limit=None):
    low_dir = os.path.join(root, "data", "LOL", split, "low")
    high_dir = os.path.join(root, "data", "LOL", split, "high")
    names = sorted(n for n in os.listdir(low_dir) if n.lower().endswith((".png", ".jpg")))
    if limit:
        names = names[:limit]
    return [(n, os.path.join(low_dir, n), os.path.join(high_dir, n)) for n in names]
