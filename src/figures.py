"""生成报告所需全部插图。"""
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ABLATION, OURS
from data import list_pairs, load_rgb
from methods import METHODS, amsr_gf
from metrics import psnr, ssim, loe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "fig")
RES = os.path.join(ROOT, "results")

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def ours_fn(cfg=None):
    c = dict(OURS if cfg is None else cfg)
    return lambda img: amsr_gf(img, **c)


# ---------- 图 1：算法流程 ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.4, 2.8))
    ax.set_xlim(0, 9.55); ax.set_ylim(0, 2.8); ax.axis("off")
    boxes = [
        (0.08, 0.95, "输入图像\n$I$", "#E8F0FE"),
        (1.66, 0.95, "引导滤波\n光照估计 $L$", "#D2E3FC"),
        (3.24, 0.95, "自适应 γ 校正\n$L'=L^{\\gamma^{*}}$", "#D2E3FC"),
        (4.82, 0.95, "反射分量\n$R=I/L'$", "#C6DAFC"),
        (6.40, 0.95, "噪声感知去噪\n+ 白平衡", "#C6DAFC"),
        (7.98, 0.95, "亮度序融合\n输出 $O$", "#AEC6F5"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.38, 1.0, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=c, ec="#4A6FA5", lw=1.0))
        ax.text(x + 0.69, y + 0.5, t, ha="center", va="center", fontsize=9.5)
    for i in range(len(boxes) - 1):
        ax.add_patch(FancyArrowPatch((boxes[i][0] + 1.40, 1.45), (boxes[i + 1][0] - 0.01, 1.45),
                                     arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#4A6FA5"))
    ax.text(4.75, 0.32, "M1 引导滤波光照估计 → M2 自适应 γ → M3 噪声感知去噪 → M4 白平衡 → M5 亮度序融合",
            ha="center", va="center", fontsize=8.2, color="#333333")
    _save(fig, "fig1_pipeline.png")


# ---------- 图 2：光照估计与中间结果 ----------
def fig_illumination(name="1.png"):
    pairs = {n: (l, h) for n, l, h in list_pairs(ROOT, "eval15")}
    low_path = os.path.join(ROOT, "data", "LOL", "eval15", "low", name)
    gt_path = os.path.join(ROOT, "data", "LOL", "eval15", "high", name)
    low, gt = load_rgb(low_path), load_rgb(gt_path)
    cfg = dict(OURS); cfg["blend"] = 0.0
    out_nofuse = amsr_gf(low, **cfg)
    out = amsr_gf(low, **OURS)
    l0 = np.max(low, axis=2)
    from guided_filter import guided_filter
    l_ms = sum(w * guided_filter(l0.astype(np.float32), l0.astype(np.float32), r, OURS["eps"])
               for r, w in zip(OURS["radii"], OURS["weights"]))
    fig, axs = plt.subplots(1, 6, figsize=(15, 3.0))
    imgs = [(low, "输入（低光照）"), (l_ms, "估计光照 $L$"), (out_nofuse, "反射分量 $R$"),
            (out, "本文输出"), (gt, "参考（正常光照）")]
    for ax, (im, t) in zip(axs, imgs):
        ax.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10); ax.axis("off")
    axs[5].hist(low.ravel(), bins=64, color="#4A6FA5", alpha=0.75, label="输入")
    axs[5].hist(gt.ravel(), bins=64, color="#E8A33D", alpha=0.55, label="参考")
    axs[5].hist(out.ravel(), bins=64, color="#5CA06B", alpha=0.45, label="输出")
    axs[5].set_title("RGB 分布", fontsize=10); axs[5].legend(fontsize=7); axs[5].tick_params(labelsize=7)
    _save(fig, "fig2_illumination.png")


# ---------- 图 3：定性对比 ----------
def fig_qualitative(names=("669.png", "146.png", "748.png"), methods=None):
    methods = methods or ["Gamma", "MSRCR", "LIME", "Ours"]
    rows = []
    for n in names:
        low = load_rgb(os.path.join(ROOT, "data", "LOL", "eval15", "low", n))
        gt = load_rgb(os.path.join(ROOT, "data", "LOL", "eval15", "high", n))
        outs = {m: (ours_fn()(low) if m == "Ours" else METHODS[m](low)) for m in methods}
        rows.append((n, low, gt, outs))
    fig, axs = plt.subplots(len(rows), len(methods) + 2, figsize=(13, 2.05 * len(rows)))
    for i, (n, low, gt, outs) in enumerate(rows):
        panels = [("输入", low)] + [(m, outs[m]) for m in methods] + [("参考", gt)]
        for j, (t, im) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(im)
            if i == 0:
                ax.set_title(t, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"场景 {i+1}", fontsize=10)
            if j > 0 and j < len(panels) - 1:
                ax.text(0.97, 0.04, f"PSNR {psnr(im, gt):.1f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6.5, color="white",
                        bbox=dict(fc="black", alpha=0.45, pad=1.0, lw=0))
            ax.set_xticks([]); ax.set_yticks([])
    _save(fig, "fig3_qualitative.png")


# ---------- 图 4：指标柱状图 ----------
def fig_metrics():
    with open(os.path.join(RES, "summary_eval15.json"), encoding="utf-8") as f:
        s = json.load(f)
    ms = ["Gamma", "CLAHE", "SSR", "MSR", "MSRCR", "LIME", "Ours"]
    labels = ["Gamma", "CLAHE", "SSR", "MSR", "MSRCR", "LIME", "本文"]
    colors = ["#9AA5B1"] * 6 + ["#D9534F"]
    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.4))
    for ax, k, t in zip(axs, ["PSNR", "SSIM", "LOE"], ["PSNR ↑ (dB)", "SSIM ↑", "LOE ↓"]):
        v = [s[m][k][0] for m in ms]; e = [s[m][k][1] for m in ms]
        ax.bar(range(len(ms)), v, yerr=e, color=colors, capsize=2, edgecolor="#333", lw=0.4)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
        ax.set_title(t, fontsize=11); ax.grid(axis="y", alpha=0.3)
        for i, x in enumerate(v):
            ax.text(i, x + e[i] + max(v) * 0.02, f"{x:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(np.array(v) + np.array(e)) * 1.18)
    _save(fig, "fig4_metrics.png")


# ---------- 图 5：消融实验 ----------
def fig_ablation(csv_path=None):
    import csv as _csv
    path = csv_path or os.path.join(RES, "ablation.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    names = [r["variant"] for r in rows]
    ps = [float(r["PSNR"]) for r in rows]
    ss = [float(r["SSIM"]) for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = ["#D9534F" if "完整" in n else "#9AA5B1" for n in names]
    axs[0].barh(range(len(names)), ps, color=cols, edgecolor="#333", lw=0.4)
    axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names, fontsize=8)
    axs[0].invert_yaxis(); axs[0].set_xlim(min(ps) - 1, max(ps) + 0.6); axs[0].set_title("PSNR ↑ (dB)", fontsize=10)
    for i, x in enumerate(ps): axs[0].text(x + 0.05, i, f"{x:.2f}", va="center", fontsize=7)
    axs[1].barh(range(len(names)), ss, color=cols, edgecolor="#333", lw=0.4)
    axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels([])
    axs[1].invert_yaxis(); axs[1].set_xlim(min(ss) - 0.05, max(ss) + 0.03); axs[1].set_title("SSIM ↑", fontsize=10)
    for i, x in enumerate(ss): axs[1].text(x + 0.004, i, f"{x:.3f}", va="center", fontsize=7)
    _save(fig, "fig5_ablation.png")


# ---------- 图 6：参数敏感性 ----------
def fig_sensitivity():
    names = [n for n, _, _ in list_pairs(ROOT, "eval15")]
    pairs = [(load_rgb(os.path.join(ROOT, "data", "LOL", "eval15", "low", n)),
              load_rgb(os.path.join(ROOT, "data", "LOL", "eval15", "high", n))) for n in names]
    targets = np.arange(0.40, 0.66, 0.05)
    ps, ss, ls = [], [], []
    for t in targets:
        cfg = dict(OURS); cfg["target"] = float(t)
        f = ours_fn(cfg)
        outs = [(f(l), g) for l, g in pairs]
        ps.append(np.mean([psnr(o, g) for o, g in outs]))
        ss.append(np.mean([ssim(o, g) for o, g in outs]))
        ls.append(np.mean([loe(o, g) for o, g in outs]))
    fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
    ax1.plot(targets, ps, "o-", color="#D9534F", label="PSNR")
    ax1.set_xlabel("目标平均亮度 τ"); ax1.set_ylabel("PSNR (dB)", color="#D9534F")
    ax2 = ax1.twinx(); ax2.plot(targets, ss, "s--", color="#4A6FA5", label="SSIM")
    ax2.set_ylabel("SSIM", color="#4A6FA5")
    ax1.grid(alpha=0.3)
    _save(fig, "fig6_sensitivity.png")


# ---------- 图 7：失败案例 ----------
def fig_failure(names=("179.png", "780.png")):
    rows = []
    for n in names:
        low = load_rgb(os.path.join(ROOT, "data", "LOL", "eval15", "low", n))
        gt = load_rgb(os.path.join(ROOT, "data", "LOL", "eval15", "high", n))
        rows.append((n, low, gt, amsr_gf(low, **OURS)))
    fig, axs = plt.subplots(len(rows), 3, figsize=(9, 2.8 * len(rows)))
    if len(rows) == 1: axs = axs[None, :]
    for i, (n, low, gt, out) in enumerate(rows):
        for ax, im, t in zip(axs[i], [low, out, gt], ["输入", "本文输出", "参考"]):
            ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis("off")
        axs[i, 1].text(0.98, 0.02, f"PSNR {psnr(out, gt):.1f}", transform=axs[i, 1].transAxes,
                       ha="right", va="bottom", fontsize=7, color="white",
                       bbox=dict(fc="black", alpha=0.45, pad=1.5, lw=0))
    _save(fig, "fig7_failure.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_illumination()
    fig_qualitative()
    fig_metrics()
    fig_ablation()
    fig_sensitivity()
    fig_failure()
