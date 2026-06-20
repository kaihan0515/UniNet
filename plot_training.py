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
    """每個類別一列（良品 + 12 種瑕疵），欄 = OK/NG 預測，全部在同一張圖。"""
    if not os.path.exists(results_csv):
        print("[confusion] 找不到 %s（請先在 GUI 按「匯出結果 CSV」）" % results_csv)
        return None
    data = []
    with open(results_csv, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            data.append(row)
    if not data:
        print("[confusion] CSV 無資料"); return None

    # 類別排序：good 在最前，其餘依代碼排序
    cats = sorted(set(r.get("category", "") for r in data),
                  key=lambda c: (c != "good", c))
    names, mat = [], np.zeros((len(cats), 2), dtype=int)   # 欄 0=OK, 1=NG
    for i, cat in enumerate(cats):
        sub = [r for r in data if r.get("category") == cat]
        mat[i, 0] = sum(1 for r in sub if r.get("prediction") == "OK")
        mat[i, 1] = sum(1 for r in sub if r.get("prediction") == "NG")
        nm = sub[0].get("category_name", "") if sub else ""
        names.append(("%s %s" % (cat, nm)).strip())

    row_sum = mat.sum(axis=1, keepdims=True)
    rate = mat / np.maximum(row_sum, 1)                    # 每列正規化（顏色深淺）

    fig, ax = plt.subplots(figsize=(6.4, 0.52 * len(cats) + 1.8))
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["OK (預測)", "NG (預測)"])
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(names)
    for i in range(len(cats)):
        for j in range(2):
            ax.text(j, i, "%d\n%.0f%%" % (mat[i, j], rate[i, j] * 100),
                    ha="center", va="center", fontsize=9,
                    color="white" if rate[i, j] > 0.5 else "black")
    # 整體抓出率 / 過殺率（good 列以外視為瑕疵）
    is_good = np.array([c == "good" for c in cats])
    tp = mat[~is_good, 1].sum(); fn = mat[~is_good, 0].sum()
    fp = mat[is_good, 1].sum(); tn = mat[is_good, 0].sum()
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0
    overkill = fp / (fp + tn) * 100 if (fp + tn) else 0
    ax.set_title("各類別預測混淆矩陣（列=實際類別）\n"
                 "瑕疵抓出率 %.1f%% / 良品過殺率 %.1f%%\n"
                 "(良品→NG=過殺；瑕疵→OK=漏檢)" % (recall, overkill), fontsize=10)
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
