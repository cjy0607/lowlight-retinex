"""消融实验：逐项关闭改进模块，观察指标变化。"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OURS
from data import list_pairs, load_rgb
from methods import amsr_gf
from metrics import psnr, ssim, loe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = [
    ("完整模型", {}),
    ("w/o M1 引导滤波", {"illum": "gaussian"}),
    ("w/o M2 自适应γ", {"adaptive_gamma": False}),
    ("w/o M3 噪声抑制", {"denoise": False}),
    ("w/o M4 白平衡", {"wb": 0.0}),
    ("w/o M5 亮度序融合", {"blend": 0.0}),
]


def main():
    pairs = [(load_rgb(l), load_rgb(h)) for _, l, h in list_pairs(ROOT, "eval15")]
    rows = []
    for name, over in VARIANTS:
        cfg = dict(OURS); cfg.update(over)
        ps, ss, ls = [], [], []
        for low, gt in pairs:
            out = amsr_gf(low, **cfg)
            ps.append(psnr(out, gt)); ss.append(ssim(out, gt)); ls.append(loe(out, gt))
        rows.append({"variant": name, "PSNR": f"{np.mean(ps):.3f}", "SSIM": f"{np.mean(ss):.4f}",
                     "LOE": f"{np.mean(ls):.3f}"})
        print(rows[-1], flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "PSNR", "SSIM", "LOE"])
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
