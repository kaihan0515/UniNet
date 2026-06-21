# -*- coding: utf-8 -*-
"""Patch 版 UniNet 訓練：從原始高解析影像對齊到 512 → 切 4×4 patch(128px)→ 放大 256,
提升小瑕疵的有效解析度。評估時把各 patch 異常圖『拼回整張』再算 image/pixel AUROC、AUPRO。

來源(原始高解析 + 遮罩)：data/steelball/steelball/
權重 -> ckpts/SteelBallPatch/steelball/，train_log.csv 同步。

執行：python train_patch.py --epochs 100 --batch_size 8 --eval_every 20 --aupro_steps 200
"""
import os, glob, copy, csv, argparse, types
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.model import UniNet
from UniNet_lib.mechanism import weighted_decision_mechanism
from eval import eval_seg_pro
from utils import setup_seed, save_weights, to_device

ROOT = r"D:/111370211/MVA/final/data/steelball/steelball"
GRID = 4; ALIGN = 512; PATCH = ALIGN // GRID; IN = 256
_to_t = T.ToTensor(); _norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_GRID_IDX = [(gy, gx) for gy in range(GRID) for gx in range(GRID)]


def detect_ball(gray):
    g = cv2.medianBlur(gray, 5); h, w = g.shape; r = min(h, w)
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, minDist=r, param1=100, param2=30,
                           minRadius=int(0.30 * r), maxRadius=int(0.55 * r))
    if cir is None:
        return w // 2, h // 2, int(0.46 * r)
    x, y, rr = np.round(cir[0, 0]).astype(int)
    return int(x), int(y), int(rr)


def align_hi(img, cx, cy, r, border, interp):
    half = int(r * 1.12); pad = half * 2
    imp = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=0)
    crop = imp[cy - half + pad:cy - half + pad + 2 * half, cx - half + pad:cx - half + pad + 2 * half]
    return cv2.resize(crop, (ALIGN, ALIGN), interpolation=interp)


def align_img(path):
    img = cv2.imread(path)
    cx, cy, r = detect_ball(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return align_hi(img, cx, cy, r, cv2.BORDER_REPLICATE, cv2.INTER_AREA), (cx, cy, r)


def patch_tensor(patch_bgr):
    rgb = cv2.cvtColor(cv2.resize(patch_bgr, (IN, IN), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return _norm(_to_t(Image.fromarray(rgb)))


def split_patches(img512):
    return [img512[gy * PATCH:(gy + 1) * PATCH, gx * PATCH:(gx + 1) * PATCH] for gy, gx in _GRID_IDX]


class GoodPatches(Dataset):
    def __init__(self, good_dir):
        self.items = []
        for ip in sorted(glob.glob(os.path.join(good_dir, "*.jpg"))):
            a, _ = align_img(ip)
            self.items += [patch_tensor(p) for p in split_patches(a)]
        print("good patches:", len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def stitch_anomaly(model, img512_bgr, c, dev):
    xs = torch.stack([patch_tensor(p) for p in split_patches(img512_bgr)]).to(dev)  # (16,3,256,256)
    ol = [[] for _ in range(model.n * 3)]
    for s in range(0, xs.shape[0], 8):                   # 分塊前向，省記憶體
        t_tf, de = model(xs[s:s + 8])
        for l, (t, ss) in enumerate(zip(t_tf, de)):
            out = 1 - F.cosine_similarity(t, ss)         # (chunk, h, w)
            for k in range(out.shape[0]):
                ol[l].append(out[k:k + 1])               # 每個 patch 一個 (1,h,w)
    _, amap = weighted_decision_mechanism(xs.shape[0], ol, c.alpha, c.beta)  # (16,256,256)
    full = np.zeros((ALIGN, ALIGN), np.float32)
    for idx, (gy, gx) in enumerate(_GRID_IDX):
        full[gy * PATCH:(gy + 1) * PATCH, gx * PATCH:(gx + 1) * PATCH] = cv2.resize(amap[idx], (PATCH, PATCH))
    return gaussian_filter(full, 4)


def build_test_list(eval_max=0):
    items = []  # (img_path, label, gt_path or None)
    for p in sorted(glob.glob(os.path.join(ROOT, "test", "good", "*.jpg"))):
        items.append((p, 0, None))
    for cat in sorted(d for d in os.listdir(os.path.join(ROOT, "test")) if d != "good"):
        for p in sorted(glob.glob(os.path.join(ROOT, "test", cat, "*.jpg"))):
            gt = os.path.join(ROOT, "ground_truth", cat, os.path.splitext(os.path.basename(p))[0] + ".png")
            items.append((p, 1, gt if os.path.exists(gt) else None))
    if eval_max:
        # 取樣加速（每類大致均勻）
        import random; random.seed(0); random.shuffle(items)
        items = items[:eval_max]
    return items


@torch.no_grad()
def evaluate(model, test_items, c, dev, aupro_steps=200):
    model.train_or_eval("eval")
    maps, gts, scores, labels = [], [], [], []
    for path, label, gt in test_items:
        a512, (cx, cy, r) = align_img(path)
        full = stitch_anomaly(model, a512, c, dev)
        m = cv2.resize(full, (IN, IN))
        if gt:
            g = cv2.imread(gt, 0)
            g = align_hi(g, cx, cy, r, cv2.BORDER_CONSTANT, cv2.INTER_NEAREST)
            g = (cv2.resize(g, (IN, IN)) > 127).astype(np.uint8)
        else:
            g = np.zeros((IN, IN), np.uint8)
        maps.append(m); gts.append(g); scores.append(float(m.max())); labels.append(label)
    maps = np.stack(maps); gts = np.stack(gts); scores = np.array(scores); labels = np.array(labels)
    img_auroc = roc_auc_score(labels, scores) * 100
    px_auroc = roc_auc_score(gts.flatten(), maps.flatten()) * 100
    pro = eval_seg_pro(gts, maps, max_step=aupro_steps)
    return img_auroc, px_auroc, pro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--aupro_steps", type=int, default=200)
    ap.add_argument("--eval_max", type=int, default=0, help="評估取樣張數(0=全部);調小可加速")
    ap.add_argument("--lr_s", type=float, default=5e-3)
    ap.add_argument("--lr_t", type=float, default=1e-6)
    args = ap.parse_args()

    setup_seed(1203)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print(dev)
    c = types.SimpleNamespace(_class_="steelball", T=2, image_size=IN, center_crop=IN,
                              weighted_decision_mechanism=True, alpha=0.01, beta=0.00003)
    ckpt = os.path.join("./ckpts", "SteelBallPatch", "steelball"); os.makedirs(ckpt, exist_ok=True)
    log_csv = os.path.join(ckpt, "train_log.csv")
    log_rows = [["epoch", "loss", "image_auroc", "pixel_auroc", "pixel_aupro"]]

    train_loader = DataLoader(GoodPatches(os.path.join(ROOT, "train", "good")),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print("steps/epoch:", len(train_loader))
    test_items = build_test_list(args.eval_max)
    print("test items:", len(test_items))

    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None; Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], dev)
    Target_teacher = copy.deepcopy(Source_teacher)
    params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr_s, betas=(0.9, 0.999), weight_decay=1e-5)
    opt1 = torch.optim.AdamW(list(Target_teacher.parameters()), lr=args.lr_t, betas=(0.9, 0.999), weight_decay=1e-5)
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    max_IRoc = max_PRoc = max_PPro = 0.0
    for epoch in range(args.epochs):
        model.train_or_eval("train")
        losses = []
        for x in train_loader:
            x = x.to(dev)
            loss = model(x)
            opt.zero_grad(); opt1.zero_grad(); loss.backward(); opt.step(); opt1.step()
            losses.append(loss.item())
        print("epoch [%d/%d] loss=%.4f" % (epoch + 1, args.epochs, float(np.mean(losses))))

        ev = (epoch + 1) % args.eval_every == 0
        isp = px = pro = 0.0
        if ev:
            isp, px, pro = evaluate(model, test_items, c, dev, args.aupro_steps)
            print("Sample Auroc: %.1f, Pixel Auroc: %.1f, Pixel Aupro: %.1f" % (isp, px, pro))
            mods = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
            if px > max_PRoc:
                max_PRoc = px; save_weights(mods, ckpt, "BEST_P_ROC")
            if pro > max_PPro:
                max_PPro = pro; save_weights(mods, ckpt, "BEST_P_PRO"); print("saved")
            max_IRoc = max(max_IRoc, isp)
            print("MAX I_ROC: %.1f, MAX P_ROC: %.1f, MAX P_PRO: %.1f" % (max_IRoc, max_PRoc, max_PPro))

        log_rows.append([epoch + 1, round(float(np.mean(losses)), 6),
                         round(isp, 4) if ev else "", round(px, 4) if ev else "", round(pro, 4) if ev else ""])
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(log_rows)

    print("done. ckpts ->", ckpt)


if __name__ == "__main__":
    main()
