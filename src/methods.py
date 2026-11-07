"""低光照增强方法集合：经典基线 + 本文改进方法 AMSR-GF。"""
import cv2
import numpy as np

try:
    from .guided_filter import guided_filter
except ImportError:  # 直接运行脚本时
    from guided_filter import guided_filter

EPS = 1e-6


# ---------- 基础工具 ----------
def to_float(img):
    return img.astype(np.float32) / 255.0


def to_u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luminance(img):
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _norm_global(x, lo_p=0.5, hi_p=99.5):
    lo, hi = np.percentile(x, [lo_p, hi_p])
    return np.clip((x - lo) / (hi - lo + EPS), 0, 1).astype(np.float32)


def _norm_per_channel(x, lo_p=0.5, hi_p=99.5):
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[2]):
        lo, hi = np.percentile(x[..., c], [lo_p, hi_p])
        out[..., c] = np.clip((x[..., c] - lo) / (hi - lo + EPS), 0, 1)
    return out


# ---------- 1. Gamma 校正 ----------
def gamma_correction(img, gamma=0.4):
    return np.clip(img ** gamma, 0, 1).astype(np.float32)


# ---------- 2. CLAHE（LAB 亮度通道） ----------
def clahe(img, clip_limit=2.0, grid=8):
    lab = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[..., 0] = op.apply(lab[..., 0])
    return to_float(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# ---------- 3. SSR ----------
def ssr(img, sigma=80):
    log_i = np.log10(img + EPS)
    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        blur = cv2.GaussianBlur(img[..., c], (0, 0), sigma)
        out[..., c] = log_i[..., c] - np.log10(blur + EPS)
    return _norm_per_channel(out)


# ---------- 4. MSR ----------
def msr(img, scales=(15, 80, 250), weights=(1 / 3, 1 / 3, 1 / 3)):
    log_i = np.log10(img + EPS)
    acc = np.zeros_like(img, dtype=np.float32)
    for s, w in zip(scales, weights):
        blur = cv2.GaussianBlur(img, (0, 0), s)
        acc += w * (log_i - np.log10(blur + EPS))
    return _norm_per_channel(acc)


# ---------- 5. MSRCR ----------
def msrcr(img, scales=(15, 80, 250), weights=(1 / 3, 1 / 3, 1 / 3),
          alpha=125.0, beta=46.0):
    log_i = np.log10(img + EPS)
    acc = np.zeros_like(img, dtype=np.float32)
    for s, w in zip(scales, weights):
        blur = cv2.GaussianBlur(img, (0, 0), s)
        acc += w * (log_i - np.log10(blur + EPS))
    # 色彩恢复因子 C_i = beta * log10(alpha * I_i + 1)
    c = beta * np.log10(alpha * img + 1.0)
    return _norm_global(c * acc, 1, 99)


# ---------- 6. LIME（引导滤波版） ----------
def lime(img, r=15, eps=1e-3, gamma=0.6):
    t = np.max(img, axis=2).astype(np.float32)
    t_ref = guided_filter(t, t, r, eps)
    t_gamma = np.clip(t_ref, 1e-4, 1.0) ** gamma
    out = img / (t_gamma[..., None] + 1e-4)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------- 7. 本文方法 AMSR-GF ----------
def _solve_gamma(lum_ratio, target):
    """二分求 gamma，使 mean(I / L^gamma) == target。lum_ratio = I_lum, L。"""
    lo, hi = 0.15, 1.50
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        m = np.mean(lum_ratio[0] / (lum_ratio[1] ** mid + 1e-4))
        if m < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gray_world(img, strength=1.0):
    """灰度世界白平衡，strength 为强度（0 表示不做）。"""
    if strength <= 0:
        return img
    m = img.reshape(-1, 3).mean(axis=0)
    gain = (m.mean() / (m + EPS)) ** strength
    return np.clip(img * gain[None, None, :], 0, 1).astype(np.float32)


def amsr_gf(img, radii=(8, 32, 96), weights=(0.5, 0.3, 0.2), eps=1e-3,
            target=0.45, gamma_range=(0.2, 1.2), denoise=True, detail=0.25,
            saturation=1.15, color_restore=True, illum="guided",
            adaptive_gamma=True, wb=0.0, blend=0.0, denoise_scale=6.0, denoise_h=None):
    """自适应多尺度引导滤波 Retinex（AMSR-GF）。"""
    img = np.clip(img, 0, 1).astype(np.float32)
    if wb > 0:
        img = gray_world(img, wb)
    l0 = np.max(img, axis=2).astype(np.float32)

    # M1: 多尺度引导滤波估计光照，替代高斯滤波，抑制光晕
    l_ms = np.zeros_like(l0)
    for r, w in zip(radii, weights):
        if illum == "gaussian":
            l_ms += w * cv2.GaussianBlur(l0, (0, 0), r / 2.0)
        else:
            l_ms += w * guided_filter(l0, l0, r, eps)
    l_ms = np.clip(l_ms, 1e-4, 1.0)

    # M2: 自适应 gamma（使输出平均亮度接近 target）
    lum = luminance(img)
    if adaptive_gamma:
        g = _solve_gamma((lum, l_ms), target)
        g = float(np.clip(g, *gamma_range))
    else:
        g = 0.35
    l_corr = np.clip(l_ms, 1e-4, 1.0) ** g

    # M4: 反射分量 + 色彩恢复因子（抑制灰化）
    r = img / (l_corr[..., None] + 1e-4)
    if color_restore:
        c = np.log10(125.0 * img + 1.0)
        c = c / (np.log10(126.0) + EPS)
        r = r * (0.5 + 0.5 * c)
    r = np.clip(r, 0, 1)

    # M3: 噪声感知抑制（暗部放大倍数大，去噪更强）
    if denoise:
        h = float(np.clip(denoise_scale * np.mean(1.0 / (l_ms + 0.05)), 2.0, 18.0))
        if denoise_h is not None:
            h = float(denoise_h)
        r = cv2.fastNlMeansDenoisingColored(to_u8(r), None, h, h, 7, 21)
        r = to_float(r)

    # M5: 细节层增强
    if detail > 0:
        blur = cv2.GaussianBlur(r, (0, 0), 1.2)
        r = np.clip(r + detail * (r - blur), 0, 1)

    # 饱和度补偿
    if saturation and saturation != 1.0:
        hsv = cv2.cvtColor(to_u8(r), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        r = to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    if blend > 0:
        base = np.clip(img ** 0.45, 0, 1)
        r = np.clip((1 - blend) * base + blend * r, 0, 1)
    return r.astype(np.float32)


METHODS = {
    "Gamma": gamma_correction,
    "CLAHE": clahe,
    "SSR": ssr,
    "MSR": msr,
    "MSRCR": msrcr,
    "LIME": lime,
    "Ours": amsr_gf,
}
