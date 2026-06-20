# -*- coding: utf-8 -*-
"""UniNet 監督式訓練 — 在(對齊後)良品上『即時合成小瑕疵 + 遮罩』,用 mask-aware
損失教模型對局部小異常反應(而非反光紋路)。在真實對齊測試集(SteelBallA)上評估。

權重 -> ckpts/SteelBallSyn/steelball/，train_log.csv 同步紀錄。
執行：
  python prepare_data/align_steelball.py        # 先有對齊資料
  python train_synthetic.py --epochs 100 --batch_size 4
"""
import os, glob, copy, csv, argparse, types
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.model import UniNet
from datasets import loading_dataset
from eval import evaluation_indusAD
from utils import setup_seed, save_weights, to_device
from synthetic_anomaly import make_synthetic

GOOD_DIR = r"D:/111370211/MVA/final/data/steelball_aligned/steelball/train/good"


class SynDataset(Dataset):
    """良品 -> 機率性合成小瑕疵；回傳 (正規化影像, 遮罩 1xHxW float 0/1)。"""
    def __init__(self, good_dir, image_size=256, p_anom=0.6):
        self.files = sorted(glob.glob(os.path.join(good_dir, "*.jpg")))
        self.size = image_size
        self.p = p_anom
        self.to_t = T.ToTensor()
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rgb = np.array(Image.open(self.files[i]).convert("RGB").resize((self.size, self.size)))
        rng = np.random.RandomState(np.random.randint(1 << 31))
        if rng.random() < self.p:
            rgb, mask = make_synthetic(rgb, rng)
        else:
            mask = np.zeros(rgb.shape[:2], np.uint8)
        x = self.norm(self.to_t(Image.fromarray(rgb)))
        m = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, m


def build_cfg(args):
    c = types.SimpleNamespace()
    c._class_ = "steelball"; c.dataset = "SteelBallA"; c.setting = "oc"; c.domain = "industrial"
    c.image_size = 256; c.center_crop = 256; c.batch_size = args.batch_size
    c.T = args.T; c.weighted_decision_mechanism = True; c.alpha = 0.01; c.beta = 0.00003
    c.is_saved = True; c.epochs = args.epochs
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--T", type=float, default=0.1, help="對比損失溫度(監督式建議 0.1)")
    ap.add_argument("--p_anom", type=float, default=0.6, help="每張圖合成瑕疵的機率")
    ap.add_argument("--lr_s", type=float, default=5e-3)
    ap.add_argument("--lr_t", type=float, default=1e-6)
    args = ap.parse_args()

    setup_seed(1203)
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(device)
    c = build_cfg(args)
    ckpt_path = os.path.join("./ckpts", "SteelBallSyn", "steelball"); os.makedirs(ckpt_path, exist_ok=True)
    log_csv = os.path.join(ckpt_path, "train_log.csv")
    log_rows = [["epoch", "loss", "image_auroc", "pixel_auroc", "pixel_aupro"]]

    train_loader = DataLoader(SynDataset(GOOD_DIR, c.image_size, args.p_anom),
                              batch_size=c.batch_size, shuffle=True, num_workers=0, drop_last=True)
    _, test_loader = loading_dataset(c, "SteelBallA")   # 真實對齊測試集

    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None; Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
    Target_teacher = copy.deepcopy(Source_teacher)
    params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr_s, betas=(0.9, 0.999), weight_decay=1e-5)
    opt1 = torch.optim.AdamW(list(Target_teacher.parameters()), lr=args.lr_t, betas=(0.9, 0.999), weight_decay=1e-5)
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    max_IRoc = max_PRoc = max_PPro = 0.0
    for epoch in range(c.epochs):
        model.train_or_eval("train")
        losses = []
        for x, m in train_loader:
            x, m = x.to(device), m.to(device)
            loss = model(x, mask=m)
            opt.zero_grad(); opt1.zero_grad(); loss.backward(); opt.step(); opt1.step()
            losses.append(loss.item())
        print("epoch [%d/%d] loss=%.4f" % (epoch + 1, c.epochs, float(np.mean(losses))))

        ev = (epoch + 1) % 10 == 0
        auroc_sp = auroc_px = aupro = 0.0
        if ev:
            auroc_px, auroc_sp, aupro = evaluation_indusAD(c, model, test_loader, device)
            print("Sample Auroc: %.1f, Pixel Auroc: %.1f, Pixel Aupro: %.1f" % (auroc_sp, auroc_px, aupro))
            modules = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
            if auroc_px > max_PRoc:
                max_PRoc = auroc_px; save_weights(modules, ckpt_path, "BEST_P_ROC")
            if aupro > max_PPro:
                max_PPro = aupro; save_weights(modules, ckpt_path, "BEST_P_PRO"); print("saved")
            max_IRoc = max(max_IRoc, auroc_sp)
            print("MAX I_ROC: %.1f, MAX P_ROC: %.1f, MAX P_PRO: %.1f" % (max_IRoc, max_PRoc, max_PPro))

        log_rows.append([epoch + 1, round(float(np.mean(losses)), 6),
                         round(float(auroc_sp), 4) if ev else "",
                         round(float(auroc_px), 4) if ev else "",
                         round(float(aupro), 4) if ev else ""])
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(log_rows)

    print("done. ckpts ->", ckpt_path)


if __name__ == "__main__":
    main()
