# -*- coding: utf-8 -*-
"""合成小瑕疵：在(對齊後、球置中的)良品影像上,於球面區域畫小的暗/亮斑點或細刮痕,
回傳 (含瑕疵影像, 遮罩)，用於 UniNet 監督式訓練,教模型對「局部小異常」反應。"""
import numpy as np
import cv2


def _ball_roi(size, frac=0.42):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size * frac), 255, -1)
    return m


def make_synthetic(rgb, rng, max_defects=3):
    """rgb: HxWx3 uint8（對齊後 256）。回傳 (含瑕疵 rgb, mask uint8 0/255)。"""
    h, w = rgb.shape[:2]
    out = rgb.copy().astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    roi = _ball_roi(h)
    ys, xs = np.where(roi > 0)
    for _ in range(rng.randint(1, max_defects + 1)):
        k = rng.randint(len(xs))
        cx, cy = int(xs[k]), int(ys[k])
        layer = np.zeros((h, w), np.uint8)
        if rng.random() < 0.5:                                   # 斑點（小黑傷/麻點/亮點）
            r = rng.randint(2, 8)
            ax = (r, max(1, int(r * rng.uniform(0.4, 1.0))))
            cv2.ellipse(layer, (cx, cy), ax, int(rng.uniform(0, 180)), 0, 360, 255, -1)
        else:                                                    # 細刮痕（刮傷/線）
            ang = rng.uniform(0, np.pi)
            L = rng.randint(8, 32)
            dx, dy = int(np.cos(ang) * L), int(np.sin(ang) * L)
            cv2.line(layer, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, rng.randint(1, 3))
        layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
        a = (layer.astype(np.float32) / 255.0)[..., None]
        delta = float(rng.randint(40, 130))
        sign = -1.0 if rng.random() < 0.7 else 1.0               # 多為暗傷,少數亮傷
        out = out + a * (sign * delta)
        mask = np.maximum(mask, (layer > 40).astype(np.uint8) * 255)
    out = np.clip(out, 0, 255).astype(np.uint8)
    mask = cv2.bitwise_and(mask, roi)                            # 只保留球內
    return out, mask
