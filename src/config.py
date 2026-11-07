"""全局实验配置（所有超参数集中在此，便于复现）。"""

OURS = dict(
    radii=(16, 48, 128),
    weights=(0.5, 0.3, 0.2),
    eps=0.05,
    target=0.60,
    gamma_range=(0.2, 1.2),
    denoise=True,
    denoise_scale=1.76,
    detail=0.0,
    saturation=1.0,
    color_restore=False,
    wb=0.5,
    blend=0.60,
)

ABLATION = {
    "w/o 引导滤波（高斯）": dict(illum="gaussian"),
    "w/o 自适应γ": dict(adaptive_gamma=False),
    "w/o 噪声抑制": dict(denoise=False),
    "w/o 白平衡": dict(wb=0.0),
    "w/o 亮度序融合": dict(blend=0.0),
    "w/o 细节增强": dict(detail=0.0),
}
