# -*- coding: utf-8 -*-
"""鋼珠影像對齊 (registration)：把每顆球置中 + 縮放到固定大小，
讓圓頂反光紋路落在一致位置，利於模型把反光學成「正常」。

來源：data/steelball/steelball/  （prepare_steelball.py 建好的 MVTec 格式）
輸出：data/steelball_aligned/steelball/  （同結構，影像與 GT 遮罩套用相同對齊）

GT 遮罩用「與其影像相同的裁切框」對齊，確保標註位置不跑掉。

執行：python prepare_data/align_steelball.py
"""
import os, glob, shutil
import numpy as np
import cv2

ROOT = r"D:/111370211/MVA/final/data/steelball/steelball"
OUT = r"D:/111370211/MVA/final/data/steelball_aligned/steelball"
OUT_SIZE = 256
MARGIN = 1.12          # 裁切框 = 2 * r * margin（球周圍留一點邊）


def detect_ball(gray):
    g = cv2.medianBlur(gray, 5)
    h, w = g.shape; r = min(h, w)
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, minDist=r, param1=100, param2=30,
                           minRadius=int(0.30 * r), maxRadius=int(0.55 * r))
    if cir is None:
        return w // 2, h // 2, int(0.46 * r)
    x, y, rr = np.round(cir[0, 0]).astype(int)
    return int(x), int(y), int(rr)


def align(img, cx, cy, r, border, interp):
    half = int(r * MARGIN); pad = half * 2
    imp = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=0)
    x0, y0 = cx - half + pad, cy - half + pad
    crop = imp[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=interp)


def reset(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def main():
    reset(OUT)
    n_img = n_mask = 0

    # train/good, test/good（無遮罩）
    for split in ["train/good", "test/good"]:
        src = os.path.join(ROOT, split)
        dst = os.path.join(OUT, split); os.makedirs(dst, exist_ok=True)
        for ip in sorted(glob.glob(os.path.join(src, "*.jpg"))):
            img = cv2.imread(ip)
            cx, cy, r = detect_ball(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            al = align(img, cx, cy, r, cv2.BORDER_REPLICATE, cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dst, os.path.basename(ip)), al)
            n_img += 1
        print(f"{split}: aligned {len(glob.glob(os.path.join(src,'*.jpg')))}")

    # test/<code> + ground_truth/<code>（影像 + 遮罩套同一裁切）
    test_dir = os.path.join(ROOT, "test")
    cats = sorted([d for d in os.listdir(test_dir)
                   if os.path.isdir(os.path.join(test_dir, d)) and d != "good"])
    for cat in cats:
        dst_t = os.path.join(OUT, "test", cat); os.makedirs(dst_t, exist_ok=True)
        dst_g = os.path.join(OUT, "ground_truth", cat); os.makedirs(dst_g, exist_ok=True)
        for ip in sorted(glob.glob(os.path.join(test_dir, cat, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            img = cv2.imread(ip)
            cx, cy, r = detect_ball(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(os.path.join(dst_t, stem + ".jpg"),
                        align(img, cx, cy, r, cv2.BORDER_REPLICATE, cv2.INTER_AREA))
            n_img += 1
            gp = os.path.join(ROOT, "ground_truth", cat, stem + ".png")
            m = cv2.imread(gp, 0) if os.path.exists(gp) else None
            if m is None:
                m = np.zeros(img.shape[:2], np.uint8)
            cv2.imwrite(os.path.join(dst_g, stem + ".png"),
                        align(m, cx, cy, r, cv2.BORDER_CONSTANT, cv2.INTER_NEAREST))
            n_mask += 1
        print(f"test/{cat}: {len(glob.glob(os.path.join(test_dir, cat, '*.jpg')))}")

    print("-" * 40)
    print(f"aligned images: {n_img}, masks: {n_mask}")
    print(f"output: {OUT}  (out_size={OUT_SIZE})")


if __name__ == "__main__":
    main()
