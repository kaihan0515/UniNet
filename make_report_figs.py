# -*- coding: utf-8 -*-
"""產生成果報告用的圖檔到 report/figs/（UniNet 分數分布、熱圖範例、背景抑制前後）。
分類器的混淆矩陣與 loss 曲線從 figs/ 複製過來。"""
import os, glob, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from visualize_steelball import build_model, anomaly_map_for, ball_roi_mask_gray, Cfg, CODE_NAMES

DATA = r"D:/111370211/MVA/final/data/steelball/steelball"
OUT = "report/figs"
os.makedirs(OUT, exist_ok=True)
DISP = 300
c = Cfg(); dev = "cuda"
model = build_model(c, "./ckpts/SteelBall/steelball", "BEST_P_PRO", dev)


def smap(ip):
    a, s = anomaly_map_for(model, c, Image.open(ip).convert("RGB"), dev)
    return gaussian_filter(a, 4), s


# 良品平均異常圖（背景基準）
gtr = sorted(glob.glob(os.path.join(DATA, "train", "good", "*.jpg")))
bg = np.mean([smap(g)[0] for g in gtr], axis=0)

# ---------- (1) 分數分布 ----------
gs = [smap(p)[1] for p in sorted(glob.glob(os.path.join(DATA, "test", "good", "*.jpg")))]
ds = []
cats = sorted([d for d in os.listdir(os.path.join(DATA, "test")) if d != "good"])
for cat in cats:
    for p in sorted(glob.glob(os.path.join(DATA, "test", cat, "*.jpg")))[:6]:
        ds.append(smap(p)[1])
gs, ds = np.array(gs), np.array(ds)
thr = gs.max() * 1.1
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gs, bins=18, alpha=0.7, label="良品 good", color="#2ecc71")
ax.hist(ds, bins=18, alpha=0.7, label="瑕疵 defect", color="#e74c3c")
ax.axvline(thr, color="k", ls="--", lw=1.2, label="門檻")
ax.set_title("UniNet 異常分數分布（良品 vs 瑕疵）")
ax.set_xlabel("anomaly score"); ax.set_ylabel("count"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "score_distribution.png"), dpi=150); plt.close(fig)
print("score range  good[%.2f,%.2f]  defect[%.2f,%.2f]" % (gs.min(), gs.max(), ds.min(), ds.max()))


def triptych(ip, gt, suppress):
    """回傳 (原圖+GT, 熱圖疊加, GT遮罩) 三張 RGB（供 matplotlib 顯示）。"""
    raw = smap(ip)[0]
    sm = np.clip(raw - bg, 0, None) if suppress else raw
    o = cv2.resize(cv2.imread(ip), (DISP, DISP))
    smD = cv2.resize(sm, (DISP, DISP))
    if suppress:
        smD = smD * ball_roi_mask_gray(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY))
    vmin, vmax = float(np.percentile(sm, 50)), float(np.percentile(sm, 99.5))
    x = np.clip((smD - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ov = cv2.addWeighted(o, 0.55, heat, 0.45, 0)
    g = cv2.resize(cv2.imread(gt, 0), (DISP, DISP)) if gt and os.path.exists(gt) else np.zeros((DISP, DISP), np.uint8)
    o2 = o.copy()
    cnts, _ = cv2.findContours((g > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o2, cnts, -1, (0, 255, 0), 2)
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb(o2), rgb(ov), cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


COLS = ["原圖 (+GT 綠框)", "異常熱圖疊加", "GT 遮罩"]


def grid_figure(rowspecs, row_labels, out_png):
    nr = len(rowspecs)
    fig, axes = plt.subplots(nr, 3, figsize=(9, 3 * nr + 0.3))
    if nr == 1:
        axes = axes[None, :]
    for r, parts in enumerate(rowspecs):
        for col in range(3):
            axes[r, col].imshow(parts[col]); axes[r, col].axis("off")
            if r == 0:
                axes[r, col].set_title(COLS[col], fontsize=12)
        axes[r, 0].text(-0.04, 0.5, row_labels[r], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=12, color="#c0392b")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


# ---------- (2) 熱圖範例（3 種瑕疵） ----------
specs, labels = [], []
for cat in ["103", "106", "107"]:
    ip = sorted(glob.glob(os.path.join(DATA, "test", cat, "*.jpg")))[0]
    stem = os.path.splitext(os.path.basename(ip))[0]
    gt = os.path.join(DATA, "ground_truth", cat, stem + ".png")
    specs.append(triptych(ip, gt, suppress=False))
    labels.append("%s %s" % (cat, CODE_NAMES.get(cat, "")))
grid_figure(specs, labels, os.path.join(OUT, "heatmap_examples.png"))

# ---------- (3) 背景抑制前後 ----------
ip = sorted(glob.glob(os.path.join(DATA, "test", "107", "*.jpg")))[0]
stem = os.path.splitext(os.path.basename(ip))[0]
gt = os.path.join(DATA, "ground_truth", "107", stem + ".png")
grid_figure([triptych(ip, gt, suppress=False), triptych(ip, gt, suppress=True)],
            ["抑制前", "抑制後 (107 生鏽)"], os.path.join(OUT, "suppress_demo.png"))

# ---------- (4) 複製分類器圖 ----------
for f in ["classifier_confusion_matrix.png", "classifier_loss_curve.png"]:
    if os.path.exists(os.path.join("figs", f)):
        shutil.copy(os.path.join("figs", f), os.path.join(OUT, f))

print("report figs ->", OUT)
print(os.listdir(OUT))
