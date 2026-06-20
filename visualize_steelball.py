# -*- coding: utf-8 -*-
"""Anomaly-heatmap visualization for the SteelBall (鋼珠) UniNet model.

Loads a trained checkpoint, runs inference on test images, and saves panels of
    [ original | anomaly heatmap overlay | ground-truth ]
per image, into ./viz/<category>/.

Usage (run from repo root, env with torch):
    python visualize_steelball.py                 # 4 imgs per defect type + some good
    python visualize_steelball.py --per_cat 6 --suffix BEST_P_PRO
"""
import os, glob, argparse, copy
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.model import UniNet
from UniNet_lib.mechanism import weighted_decision_mechanism
from utils import load_weights, to_device

DATA = r"D:/111370211/MVA/final/data/steelball/steelball"
OUT  = "./viz"
DISP = 320  # display size of each panel tile

CODE_NAMES = {
    "100": "小黑傷", "101": "灰傷刻痕", "102": "麻點", "103": "大黑傷",
    "104": "研磨傷", "105": "肯傷", "106": "刮傷", "107": "生鏽",
    "108": "霧面", "109": "亮傷-暗", "110": "小白點線", "111": "亮傷-亮",
}


class Cfg:
    """Minimal config mirroring the fields main.py sets for the one-class path."""
    dataset = "SteelBall"; setting = "oc"; domain = "industrial"
    _class_ = "steelball"
    image_size = 256; center_crop = 256; batch_size = 1
    T = 2
    weighted_decision_mechanism = True
    alpha = 0.01; beta = 0.00003


def build_model(c, ckpt_path, suffix, device):
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
    Target_teacher = copy.deepcopy(Source_teacher)
    new_state = load_weights([Target_teacher, bn, student, DFS], ckpt_path, suffix)
    Target_teacher, bn, student, DFS = new_state['tt'], new_state['bn'], new_state['st'], new_state['dfs']
    model = UniNet(c, Source_teacher.to(device).eval(), Target_teacher, bn, student, DFS)
    model.train_or_eval(type='eval')
    return model


def anomaly_map_for(model, c, pil_img, device):
    tf = T.Compose([T.Resize((c.image_size, c.image_size), InterpolationMode.LANCZOS), T.ToTensor()])
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = norm(tf(pil_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        t_tf, de_features = model(x)
        output_list = [[] for _ in range(model.n * 3)]
        for l, (t, s) in enumerate(zip(t_tf, de_features)):
            output_list[l].append(1 - F.cosine_similarity(t, s))
        score, amap = weighted_decision_mechanism(1, output_list, c.alpha, c.beta)
    return amap[0], float(np.array(score).reshape(-1)[0])  # (256,256), scalar score


def ball_roi_mask_gray(gray, shrink=0.92):
    """偵測鋼珠圓形，回傳 {0,1} float mask（球外=0）。"""
    g = cv2.medianBlur(gray, 5)
    h, w = g.shape; r = min(h, w)
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, minDist=r, param1=100, param2=30,
                           minRadius=int(0.30 * r), maxRadius=int(0.55 * r))
    if cir is None:
        cx, cy, rad = w // 2, h // 2, int(0.46 * r)
    else:
        cx, cy, rad = np.round(cir[0, 0]).astype(int)
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (int(cx), int(cy)), int(rad * shrink), 1.0, -1)
    return mask


def overlay_panel(img_path, sm, gt_path, vmin, vmax, suppress=False):
    """sm: 已平滑(且若 suppress 已減背景)的 256x256 異常圖。"""
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (DISP, DISP))
    am = cv2.resize(sm, (DISP, DISP))
    if suppress:
        am = am * ball_roi_mask_gray(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY))
    am = (am - vmin) / (vmax - vmin + 1e-8)
    am = np.clip(am, 0, 1)
    heat = cv2.applyColorMap((am * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.55, heat, 0.45, 0)

    if gt_path and os.path.exists(gt_path):
        gt = cv2.resize(cv2.imread(gt_path, 0), (DISP, DISP))
        gt_vis = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig, cnts, -1, (0, 255, 0), 2)      # GT outline on original (green)
    else:
        gt_vis = np.zeros((DISP, DISP, 3), np.uint8)

    panel = cv2.hconcat([orig, overlay, gt_vis])
    for i, txt in enumerate(["original (+GT)", "anomaly heatmap", "ground truth"]):
        cv2.putText(panel, txt, (10 + i * DISP, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cat", type=int, default=4, help="images per defect category")
    ap.add_argument("--good", type=int, default=4, help="good/OK images to also visualize")
    ap.add_argument("--suffix", default="BEST_P_PRO")
    ap.add_argument("--suppress", action="store_true",
                    help="抑制背景/反光：減良品平均圖 + 遮掉鋼珠圓形外")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = Cfg()
    ckpt_path = os.path.join("./ckpts", "SteelBall", "steelball")
    model = build_model(c, ckpt_path, args.suffix, device)
    print("model loaded from", os.path.join(ckpt_path, args.suffix + ".pth"))

    cats = sorted([d for d in os.listdir(os.path.join(DATA, "test"))])
    os.makedirs(OUT, exist_ok=True)

    # 抑制模式：先用良品 train/good 算平均異常圖當背景基準
    bg_map = None
    if args.suppress:
        goods = sorted(glob.glob(os.path.join(DATA, "train", "good", "*.jpg")))
        acc = None
        for g in goods:
            am, _ = anomaly_map_for(model, c, Image.open(g).convert("RGB"), device)
            sm = gaussian_filter(am, sigma=4)
            acc = sm if acc is None else acc + sm
        bg_map = acc / max(len(goods), 1)
        print(f"suppress on: bg_map built from {len(goods)} good images")

    # ---- pass 1: compute all anomaly maps (smoothed, suppressed), collect GLOBAL scale ----
    items = []          # (cat, img_path, gt_path, sm, score)
    pooled = []
    for cat in cats:
        imgs = sorted(glob.glob(os.path.join(DATA, "test", cat, "*.jpg")))
        n = args.good if cat == "good" else args.per_cat
        for ip in imgs[:n]:
            stem = os.path.splitext(os.path.basename(ip))[0]
            amap, score = anomaly_map_for(model, c, Image.open(ip).convert("RGB"), device)
            sm = gaussian_filter(amap, sigma=4)
            if args.suppress and bg_map is not None:
                sm = np.clip(sm - bg_map, 0, None)
            gt = None if cat == "good" else os.path.join(DATA, "ground_truth", cat, stem + ".png")
            items.append((cat, ip, gt, sm, score))
            pooled.append(sm.ravel())
    pooled = np.concatenate(pooled)
    # shared scale: good imgs stay cool, defect hot-spots go red (robust percentiles)
    vmin, vmax = float(np.percentile(pooled, 50)), float(np.percentile(pooled, 99.5))

    # ---- pass 2: render with the shared scale ----
    per_cat_cnt = {}
    for cat, ip, gt, sm, score in items:
        out_dir = os.path.join(OUT, cat)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(ip))[0]
        panel = overlay_panel(ip, sm, gt, vmin, vmax, suppress=args.suppress)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_score{score:.3f}.png"), panel)
        per_cat_cnt[cat] = per_cat_cnt.get(cat, 0) + 1
    for cat, cnt in sorted(per_cat_cnt.items()):
        print(f"  {cat} ({CODE_NAMES.get(cat, cat)}): {cnt} panels")
    print(f"global color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")
    print("done. panels saved under", OUT)


if __name__ == "__main__":
    main()
