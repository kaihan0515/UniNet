# -*- coding: utf-8 -*-
"""
多類別鋼珠瑕疵『分類器』（13 類：Normal + 12 種瑕疵）。

與 UniNet 的「異常偵測(OK/NG)」不同，這支是真正的多類別分類，因此能輸出
標準的 N×N 正規化混淆矩陣（True label vs Predicted label）。

資料：
  Normal   <- data/OK 2/
  100..111 <- data/steel ball dataset/<代碼...>/   （每個資料夾一類）

用法（從 UniNet repo 根目錄，環境 MVA_py310_cu121）：
  python train_classifier.py
  python train_classifier.py --epochs 40 --batch 32 --img_size 224

輸出：
  figs/classifier_confusion_matrix.png   正規化混淆矩陣
  figs/classifier_report.txt             per-class precision/recall/f1
  ckpts/classifier/steelball_resnet18.pth
"""
import os, glob, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                           "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = r"D:/111370211/MVA/final/data"
NORMAL_DIR = os.path.join(DATA_ROOT, "OK 2")
DEFECT_ROOT = os.path.join(DATA_ROOT, "steel ball dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CODE_NAMES = {
    "100": "小黑傷", "101": "灰傷刻痕", "102": "麻點", "103": "大黑傷",
    "104": "研磨傷", "105": "肯傷", "106": "刮傷", "107": "生鏽",
    "108": "霧面", "109": "亮傷-暗", "110": "小白點線", "111": "亮傷-亮",
}


def gather():
    items = []
    for f in glob.glob(os.path.join(NORMAL_DIR, "*.jpg")):
        items.append((f, "Normal"))
    for d in sorted(os.listdir(DEFECT_ROOT)):
        full = os.path.join(DEFECT_ROOT, d)
        if not os.path.isdir(full) or d == "good":
            continue
        code = "".join(ch for ch in d if ch.isdigit())[:3] or d[:3]
        for f in glob.glob(os.path.join(full, "*.jpg")):
            items.append((f, code))
    return items


class DS(Dataset):
    def __init__(self, items, cls2idx, tf):
        self.items = items; self.cls2idx = cls2idx; self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), self.cls2idx[c]


def plot_cm(ys, ps, classes, disp, out_png):
    cm = confusion_matrix(ys, ps, labels=list(range(len(classes))))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.85 * n + 3, 0.85 * n + 2.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(disp, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(disp)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "%.2f" % cmn[i, j], ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(BASE, "figs"))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(BASE, "ckpts", "classifier"); os.makedirs(ckpt_dir, exist_ok=True)

    items = gather()
    codes = sorted(set(c for _, c in items if c != "Normal"))
    classes = ["Normal"] + codes
    disp = ["Normal"] + [CODE_NAMES.get(c, c) for c in codes]
    cls2idx = {c: i for i, c in enumerate(classes)}
    labels = [c for _, c in items]
    tr, te = train_test_split(items, test_size=0.2, stratify=labels, random_state=args.seed)
    print("classes=%d  total=%d  train=%d  test=%d  device=%s"
          % (len(classes), len(items), len(tr), len(te), DEVICE))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf_tr = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_te = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    # num_workers=0：此機器 Windows DataLoader 多進程會死鎖
    dl_tr = DataLoader(DS(tr, cls2idx, tf_tr), batch_size=args.batch, shuffle=True, num_workers=0)
    dl_te = DataLoader(DS(te, cls2idx, tf_te), batch_size=args.batch, shuffle=False, num_workers=0)

    cnt = np.bincount([cls2idx[c] for _, c in tr], minlength=len(classes))
    w = torch.tensor(cnt.sum() / (len(classes) * np.maximum(cnt, 1)), dtype=torch.float32).to(DEVICE)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(weight=w)

    losses = []
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        losses.append(tot / len(tr))
        if ep == 0 or (ep + 1) % 5 == 0:
            print("epoch %d/%d  loss=%.4f" % (ep + 1, args.epochs, losses[-1]))

    # loss 曲線
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3, lw=1.5, color="#2980b9")
    ax.set_title("分類器訓練 Loss 曲線"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); fig.tight_layout()
    loss_png = os.path.join(args.out, "classifier_loss_curve.png")
    fig.savefig(loss_png, dpi=150); plt.close(fig)
    print("saved ->", loss_png)

    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl_te:
            pred = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            ps += pred.tolist(); ys += y.numpy().tolist()
    acc = accuracy_score(ys, ps) * 100
    print("test accuracy = %.2f%%" % acc)
    report = classification_report(ys, ps, target_names=disp, zero_division=0)
    print(report)

    out_png = os.path.join(args.out, "classifier_confusion_matrix.png")
    plot_cm(ys, ps, classes, disp, out_png)
    with open(os.path.join(args.out, "classifier_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("test accuracy = %.2f%%\n\n%s\n" % (acc, report))
    torch.save({"state_dict": model.state_dict(), "classes": classes, "disp": disp},
               os.path.join(ckpt_dir, "steelball_resnet18.pth"))
    print("saved ->", out_png)
    print("saved ->", os.path.join(ckpt_dir, "steelball_resnet18.pth"))


if __name__ == "__main__":
    main()
