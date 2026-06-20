# -*- coding: utf-8 -*-
"""
畫 UniNet 鋼珠檢測的訓練/評估圖（給報告用）：

  1) loss 曲線（+ image/pixel AUROC、AUPRO 曲線）
        來源：ckpts/SteelBall/steelball/train_log.csv
        （由 train_unsupervisedAD.py 在訓練時自動產生；舊模型若無此檔，需重新訓練一次）

  2) 混淆矩陣（+ 分數分布）
        來源：GUI 的「匯出結果 CSV」(uninet_results.csv)

用法：
    python plot_training.py                               # 兩張都畫（用預設路徑）
    python plot_training.py --log ckpts/SteelBall/steelball/train_log.csv
    python plot_training.py --results uninet_results.csv --out figs
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                           "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(BASE, "ckpts", "SteelBall", "steelball", "train_log.csv")
DEFAULT_RESULTS = os.path.join(BASE, "uninet_results.csv")


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def plot_loss(log_csv, out_dir):
    if not os.path.exists(log_csv):
        print("[loss] 找不到 %s\n      （loss 曲線需要訓練時的 train_log.csv；"
              "舊模型沒有此檔，請用最新的 train_unsupervisedAD.py 重新訓練一次。）" % log_csv)
        return None
    epochs, loss, iroc, proc, ppro = [], [], [], [], []
    with open(log_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            e = _to_float(row.get("epoch"))
            if e is None:
                continue
            epochs.append(e); loss.append(_to_float(row.get("loss")))
            iroc.append(_to_float(row.get("image_auroc")))
            proc.append(_to_float(row.get("pixel_auroc")))
            ppro.append(_to_float(row.get("pixel_aupro")))
    if not epochs:
        print("[loss] %s 沒有資料列" % log_csv)
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    ax1.plot(epochs, loss, color="#2980b9", marker="o", ms=2, lw=1.4)
    ax1.set_title("訓練 Loss 曲線"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)

    def _series(vals):
        xs = [e for e, v in zip(epochs, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        return xs, ys
    for vals, lab, col in [(iroc, "image AUROC", "#27ae60"),
                           (proc, "pixel AUROC", "#e67e22"),
                           (ppro, "pixel AUPRO", "#c0392b")]:
        xs, ys = _series(vals)
        if xs:
            ax2.plot(xs, ys, marker="o", ms=4, lw=1.4, label=lab, color=col)
    ax2.set_title("評估指標（每 10 epoch）"); ax2.set_xlabel("epoch"); ax2.set_ylabel("%")
    ax2.set_ylim(0, 101); ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("[loss] saved ->", p)
    return p


def plot_confusion(results_csv, out_dir):
    if not os.path.exists(results_csv):
        print("[confusion] 找不到 %s（請先在 GUI 按「匯出結果 CSV」）" % results_csv)
        return None
    labels, preds, scores, thr = [], [], [], None
    with open(results_csv, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            labels.append(1 if row.get("label") == "defect" else 0)
            preds.append(1 if row.get("prediction") == "NG" else 0)
            scores.append(_to_float(row.get("anomaly_score")))
            thr = _to_float(row.get("threshold"))
    labels = np.array(labels); preds = np.array(preds); scores = np.array(scores, dtype=float)
    if len(labels) == 0:
        print("[confusion] CSV 無資料"); return None

    tn = int(((preds == 0) & (labels == 0)).sum()); fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum()); tp = int(((preds == 1) & (labels == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0
    overkill = fp / (fp + tn) * 100 if (fp + tn) else 0

    fig, (axc, axh) = plt.subplots(1, 2, figsize=(11, 4.6))
    axc.imshow(cm, cmap="Blues")
    axc.set_xticks([0, 1]); axc.set_xticklabels(["OK (預測)", "NG (預測)"])
    axc.set_yticks([0, 1]); axc.set_yticklabels(["良品 (實際)", "瑕疵 (實際)"])
    half = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            axc.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > half else "black", fontsize=15)
    axc.set_title("混淆矩陣  (抓出率 %.1f%% / 過殺率 %.1f%%)" % (recall, overkill))

    good = scores[labels == 0]; bad = scores[labels == 1]
    if len(good):
        axh.hist(good, bins=30, alpha=0.6, label="良品 good", color="#2ecc71")
    if len(bad):
        axh.hist(bad, bins=30, alpha=0.6, label="瑕疵 defect", color="#e74c3c")
    if thr is not None:
        axh.axvline(thr, color="k", ls="--", lw=1.2, label="門檻 %.3f" % thr)
    axh.set_title("分數分布"); axh.set_xlabel("anomaly score"); axh.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("[confusion] saved ->", p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG, help="train_log.csv 路徑")
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="GUI 匯出的結果 CSV 路徑")
    ap.add_argument("--out", default=os.path.join(BASE, "figs"), help="圖檔輸出資料夾")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_loss(args.log, args.out)
    plot_confusion(args.results, args.out)
    print("done. 圖檔在", args.out)


if __name__ == "__main__":
    main()
