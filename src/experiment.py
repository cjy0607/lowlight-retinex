"""批量实验：在 LOLv1 测试集上运行全部方法并保存指标。"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OURS
from data import list_pairs, load_rgb, to_float
from methods import METHODS, amsr_gf
from metrics import METRIC_FUNCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 所有对比方法（Ours 使用固定配置）
ALL_METHODS = ["Gamma", "CLAHE", "SSR", "MSR", "MSRCR", "LIME", "Ours"]


def make_ours(cfg=None):
    cfg = dict(OURS if cfg is None else cfg)

    def fn(img):
        return amsr_gf(img, **cfg)

    return fn


def run(split="eval15", methods=None, cfg=None, tag="eval15", verbose=True):
    methods = methods or ALL_METHODS
    funcs = {m: METHODS[m] for m in methods if m != "Ours"}
    if "Ours" in methods:
        funcs["Ours"] = make_ours(cfg)
    pairs = list_pairs(ROOT, split)
    rows = []
    for i, (name, lp, hp) in enumerate(pairs):
        low, gt = load_rgb(lp), load_rgb(hp)
        for m in methods:
            t0 = time.perf_counter()
            out = funcs[m](low)
            dt = time.perf_counter() - t0
            rec = {"image": name, "method": m, "time": dt}
            for k, f in METRIC_FUNCS.items():
                rec[k] = f(out, gt) if k in ("PSNR", "SSIM", "LOE") else f(out)
            rows.append(rec)
        if verbose:
            print(f"[{i + 1}/{len(pairs)}] {name}", flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    per_path = os.path.join(ROOT, "results", f"per_image_{tag}.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        summary[m] = {
            k: [float(np.mean([r[k] for r in sub])), float(np.std([r[k] for r in sub]))]
            for k in list(METRIC_FUNCS) + ["time"]
        }
    with open(os.path.join(ROOT, "results", f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "results", f"summary_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{k}_mean" for k in summary[methods[0]]] + [f"{k}_std" for k in summary[methods[0]]])
        for m in methods:
            w.writerow([m] + [f"{summary[m][k][0]:.4f}" for k in summary[m]] + [f"{summary[m][k][1]:.4f}" for k in summary[m]])

    if verbose:
        print("\n%-8s %-6s %-6s %-6s %-6s %-6s %-6s %-8s" % ("method", "PSNR", "SSIM", "LOE", "Entropy", "Gradient", "Colorful", "Time(ms)"))
        for m in methods:
            s = summary[m]
            print("%-8s %-6.2f %-6.3f %-6.2f %-6.2f %-6.3f %-6.3f %-8.0f" % (
                m, s["PSNR"][0], s["SSIM"][0], s["LOE"][0], s["Entropy"][0],
                s["Gradient"][0], s["Colorful"][0], s["time"][0] * 1000))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval15")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.split, tag=args.tag or args.split)
