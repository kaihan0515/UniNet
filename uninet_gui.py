# -*- coding: utf-8 -*-
"""
UniNet 鋼珠檢測 — 成效檢視 GUI  (PyQt5, MVC 架構)

Model      : UniNetADModel — 載入訓練好的 UniNet (source/target teacher + bn +
             student + DFS) 權重，提供 predict(影像) -> (異常分數, 熱圖)；
             啟動時用 train/good 計算良品分數分布 -> 預設門檻。
View       : Qt 視窗 — 原圖 + 異常熱圖疊加、OK/NG、分數、門檻滑桿、
             AUC/ROC 與分數分布圖。
Controller : MainWindow 的 slot / 背景執行緒（載入、批次評估）。

執行（從 UniNet repo 根目錄）：
    conda activate MVA_py310_cu121
    python uninet_gui.py
"""
import os
import sys
import glob
import copy
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from scipy.ndimage import gaussian_filter

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                           "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

import cv2

# UniNet internals
from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.model import UniNet
from UniNet_lib.mechanism import weighted_decision_mechanism
from utils import load_weights, to_device

# --------------------------------------------------------------------------- #
# 預設路徑（可在 GUI 內用按鈕更換）。資料在 repo 外的 ../data。
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_DIR = os.path.join(BASE, "ckpts", "SteelBall", "steelball")
DEFAULT_TRAIN_GOOD = os.path.normpath(os.path.join(
    BASE, "..", "data", "steelball", "steelball", "train", "good"))
DEFAULT_TEST_DIR = os.path.normpath(os.path.join(
    BASE, "..", "data", "steelball", "steelball", "test"))

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CODE_NAMES = {
    "100": "小黑傷", "101": "灰傷刻痕", "102": "麻點", "103": "大黑傷",
    "104": "研磨傷", "105": "肯傷", "106": "刮傷", "107": "生鏽",
    "108": "霧面", "109": "亮傷-暗", "110": "小白點線", "111": "亮傷-亮",
    "good": "良品",
}


def list_images(folder):
    files = []
    for e in IMG_EXTS:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def make_overlay_bgr(pil, disp):
    """原圖(BGR) 疊上 JET 熱圖。disp 為 0–255 的熱圖，回傳 BGR np.uint8。"""
    bgr = np.array(pil)[:, :, ::-1].copy()
    heat = cv2.applyColorMap(np.clip(disp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)


def ball_roi_mask(pil, shrink=0.92):
    """偵測鋼珠圓形範圍，回傳 {0,1} float mask（球外=0，用來遮掉背景）。"""
    g = cv2.medianBlur(np.array(pil.convert("L")), 5)
    h, w = g.shape; r = min(h, w)
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, minDist=r, param1=100, param2=30,
                           minRadius=int(0.30 * r), maxRadius=int(0.55 * r))
    if cir is None:
        cx, cy, rad = w // 2, h // 2, int(0.46 * r)          # 偵測失敗 -> 置中圓
    else:
        cx, cy, rad = np.round(cir[0, 0]).astype(int)
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (int(cx), int(cy)), int(rad * shrink), 1.0, -1)
    return mask


# ===========================================================================
#  MODEL
# ===========================================================================
class Cfg:
    """UniNet 一類 (one-class) 推論所需的最小設定，對齊 main.py 預設。"""
    dataset = "SteelBall"; setting = "oc"; domain = "industrial"
    _class_ = "steelball"
    image_size = 256; center_crop = 256; batch_size = 1
    T = 2
    weighted_decision_mechanism = True
    alpha = 0.01; beta = 0.00003


class UniNetADModel:
    """封裝訓練好的 UniNet，負責推論與門檻設定。"""

    def __init__(self, weights_dir, suffix="BEST_P_PRO", device=DEVICE):
        self.weights_dir = weights_dir
        self.suffix = suffix
        self.device = device
        self.c = Cfg()
        self.model = None
        self.train_good_scores = None
        self.default_threshold = None
        self.bg_map = None                 # 良品平均異常圖（背景/反光基準，供抑制用）
        self._tf = T.Compose([
            T.Resize((self.c.image_size, self.c.image_size), InterpolationMode.LANCZOS),
            T.ToTensor()])
        self._norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def is_ready(self):
        return self.model is not None

    # --- 載入權重（複刻 test.py 的 un_cls 流程）--------------------------- #
    def load(self):
        c = self.c
        Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
        Source_teacher.layer4 = None
        Source_teacher.fc = None
        student = de_wide_resnet50_2(pretrained=False)
        DFS = DomainRelated_Feature_Selection()
        [Source_teacher, bn, student, DFS] = to_device(
            [Source_teacher, bn, student, DFS], self.device)
        Target_teacher = copy.deepcopy(Source_teacher)
        ns = load_weights([Target_teacher, bn, student, DFS], self.weights_dir, self.suffix)
        Target_teacher, bn, student, DFS = ns['tt'], ns['bn'], ns['st'], ns['dfs']
        self.model = UniNet(c, Source_teacher.to(self.device).eval(),
                            Target_teacher, bn, student, DFS)
        self.model.train_or_eval(type='eval')

    @torch.no_grad()
    def _infer(self, pil_image):
        """前向推論，回傳 (異常分數 float, 原始 256x256 異常圖 np.float32)。"""
        x = self._norm(self._tf(pil_image))[None].to(self.device)
        t_tf, de_features = self.model(x)
        output_list = [[] for _ in range(self.model.n * 3)]
        for l, (t, s) in enumerate(zip(t_tf, de_features)):
            output_list[l].append(1 - F.cosine_similarity(t, s))
        score, amap = weighted_decision_mechanism(1, output_list, self.c.alpha, self.c.beta)
        return float(np.array(score).reshape(-1)[0]), amap[0]

    # --- 單張推論（suppress_bg=True 抑制背景/反光；分數不受影響）---------- #
    @torch.no_grad()
    def predict(self, pil_image, suppress_bg=False):
        ow, oh = pil_image.size
        score, amap = self._infer(pil_image)
        m = gaussian_filter(amap, sigma=4)                  # 256x256
        if suppress_bg and self.bg_map is not None:
            m = np.clip(m - self.bg_map, 0, None)           # 減良品平均反光響應
        disp = cv2.resize(m, (ow, oh))
        if suppress_bg:
            disp = disp * ball_roi_mask(pil_image)          # 遮掉鋼珠圓形外的背景
        mn, mx = float(disp.min()), float(disp.max())
        disp = (disp - mn) / (mx - mn + 1e-8) * 255.0       # min-max 正規化到 0–255
        return score, disp.astype(np.float32)

    @torch.no_grad()
    def predicted_mask(self, pil_image, pct=99.0):
        """UniNet 預測遮罩：減背景殘差在球內取自身高百分位二值化（uint8 0/255）。"""
        ow, oh = pil_image.size
        _, amap = self._infer(pil_image)
        sm = gaussian_filter(amap, sigma=4)
        if self.bg_map is not None:
            sm = np.clip(sm - self.bg_map, 0, None)
        sub = cv2.resize(sm, (ow, oh))
        roi = ball_roi_mask(pil_image)
        sub = sub * roi
        vals = sub[roi > 0.5]
        t = float(np.percentile(vals, pct)) if vals.size else 0.0
        return (((sub > t) & (roi > 0.5)).astype(np.uint8) * 255)

    # --- 用良品集算分數 -> 預設門檻 + 平均異常圖(背景基準) --------------- #
    @torch.no_grad()
    def compute_train_scores(self, good_dir, progress=None):
        files = list_images(good_dir)
        if not files:
            raise RuntimeError("找不到良品影像：%s" % good_dir)
        scores, acc = [], None
        for i, f in enumerate(files):
            sc, amap = self._infer(Image.open(f).convert("RGB"))
            scores.append(sc)
            gm = gaussian_filter(amap, sigma=4)
            acc = gm if acc is None else acc + gm
            if progress:
                progress(i + 1, len(files), "計算良品分數/背景基準")
        self.train_good_scores = np.array(scores, dtype=np.float32)
        self.default_threshold = float(self.train_good_scores.max() * 1.1)
        self.bg_map = acc / len(files)                       # 256x256 良品平均異常圖


# ===========================================================================
#  背景執行緒
# ===========================================================================
class LoadWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(bool, str)

    def __init__(self, model, good_dir):
        super().__init__()
        self.model = model
        self.good_dir = good_dir

    def run(self):
        try:
            self.progress.emit(0, 1, "載入模型權重")
            self.model.load()
            self.model.compute_train_scores(
                self.good_dir,
                progress=lambda i, n, t: self.progress.emit(i, n, t))
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))


class EvalWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(object, object, object, object)   # paths, labels, scores, cats

    def __init__(self, model, test_dir):
        super().__init__()
        self.model = model
        self.test_dir = test_dir

    def run(self):
        paths, labels, cats = [], [], []
        for sub in sorted(os.listdir(self.test_dir)):
            d = os.path.join(self.test_dir, sub)
            if not os.path.isdir(d):
                continue
            label = 0 if sub.lower() == "good" else 1
            for f in list_images(d):
                paths.append(f)
                labels.append(label)
                cats.append(sub)                       # 子資料夾名 = 瑕疵類別代碼
        scores = []
        for i, p in enumerate(paths):
            sc, _ = self.model.predict(Image.open(p).convert("RGB"))
            scores.append(sc)
            self.progress.emit(i + 1, len(paths), "評估測試集")
        self.done.emit(paths, np.array(labels), np.array(scores, dtype=np.float32), cats)


class SaveHeatmapWorker(QtCore.QThread):
    """批次推論並另存 [原圖 | 熱圖疊加] 圖檔（依類別分子資料夾）。"""
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(int, str)                 # 已存張數, 輸出資料夾

    def __init__(self, model, items, out_dir, threshold, suppress_bg=False):
        super().__init__()
        self.model = model
        self.items = items                             # list of (path, cat)
        self.out_dir = out_dir
        self.threshold = threshold
        self.suppress_bg = suppress_bg

    def run(self):
        n, cnt = len(self.items), 0
        for i, (p, cat) in enumerate(self.items):
            try:
                pil = Image.open(p).convert("RGB")
                score, disp = self.model.predict(pil, suppress_bg=self.suppress_bg)
                orig = np.array(pil)[:, :, ::-1].copy()           # BGR
                overlay = make_overlay_bgr(pil, disp)
                panel = cv2.hconcat([orig, overlay])
                is_ng = (self.threshold is not None) and (score > self.threshold)
                txt = "%s  score=%.3f" % ("NG" if is_ng else "OK", score)
                color = (0, 0, 255) if is_ng else (0, 200, 0)
                cv2.putText(panel, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                d = os.path.join(self.out_dir, cat)
                os.makedirs(d, exist_ok=True)
                stem = os.path.splitext(os.path.basename(p))[0]
                cv2.imwrite(os.path.join(d, "%s_heat.png" % stem), panel)
                cnt += 1
            except Exception:
                pass
            self.progress.emit(i + 1, n, "另存熱圖")
        self.done.emit(cnt, self.out_dir)


# ===========================================================================
#  VIEW + CONTROLLER
# ===========================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniNet 鋼珠檢測 — 成效檢視")
        self.resize(1480, 820)

        self.model = UniNetADModel(DEFAULT_WEIGHTS_DIR)
        self.weights_dir = DEFAULT_WEIGHTS_DIR
        self.train_good = DEFAULT_TRAIN_GOOD
        self.test_dir = DEFAULT_TEST_DIR
        self.threshold = None
        self.cur_image_path = None
        self.cur_disp_map = None
        self.cur_score = None
        self.eval_paths = self.eval_labels = self.eval_scores = self.eval_cats = None

        self._build_ui()
        self._start_load()

    # ---------------- UI ---------------- #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ===== 顯示區（左）：影像 + 判定 =====
        disp = QtWidgets.QVBoxLayout()
        root.addLayout(disp, 2)
        img_grid = QtWidgets.QGridLayout()
        self.lbl_orig = self._image_label("原始影像", 240)
        self.lbl_heat = self._image_label("異常熱圖疊加", 240)
        self.lbl_pred = self._image_label("UniNet 預測遮罩", 240)
        self.lbl_mask = self._image_label("瑕疵遮罩 (GT)", 240)
        img_grid.addWidget(self.lbl_orig[0], 0, 0)   # 上左：原圖
        img_grid.addWidget(self.lbl_heat[0], 0, 1)   # 上右：熱圖
        img_grid.addWidget(self.lbl_pred[0], 1, 0)   # 下左：預測遮罩
        img_grid.addWidget(self.lbl_mask[0], 1, 1)   # 下右：GT 遮罩
        disp.addLayout(img_grid, 1)

        verdict_box = QtWidgets.QGroupBox("判定結果")
        v = QtWidgets.QVBoxLayout(verdict_box)
        self.lbl_verdict = QtWidgets.QLabel("—")
        self.lbl_verdict.setAlignment(QtCore.Qt.AlignCenter)
        f = self.lbl_verdict.font(); f.setPointSize(26); f.setBold(True)
        self.lbl_verdict.setFont(f)
        self.lbl_score = QtWidgets.QLabel("異常分數: —    門檻: —")
        self.lbl_score.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.lbl_verdict)
        v.addWidget(self.lbl_score)
        disp.addWidget(verdict_box)

        # ===== 圖表區（中）：分數分布 / ROC + 指標 =====
        charts = QtWidgets.QVBoxLayout()
        root.addLayout(charts, 2)
        self.canvas = FigureCanvas(Figure(figsize=(4.2, 6)))
        self.ax_hist = self.canvas.figure.add_subplot(211)
        self.ax_roc = self.canvas.figure.add_subplot(212)
        self._init_axes()
        charts.addWidget(self.canvas, 1)
        self.lbl_metrics = QtWidgets.QLabel("尚未評估測試集")
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setAlignment(QtCore.Qt.AlignTop)
        charts.addWidget(self.lbl_metrics)

        # ===== 操作區（右）：所有「需點選」的控制集中於此 =====
        ctrl_box = QtWidgets.QGroupBox("操作區")
        ctrl_box.setFixedWidth(300)
        ctrl = QtWidgets.QVBoxLayout(ctrl_box)
        self.btn_open_img = QtWidgets.QPushButton("開啟單張影像")
        self.btn_open_folder = QtWidgets.QPushButton("開啟資料夾")
        ctrl.addWidget(self.btn_open_img)
        ctrl.addWidget(self.btn_open_folder)
        ctrl.addWidget(QtWidgets.QLabel("影像清單"))
        self.list_widget = QtWidgets.QListWidget()
        ctrl.addWidget(self.list_widget, 1)
        self.btn_eval = QtWidgets.QPushButton("評估測試集（AUC / 分布）")
        ctrl.addWidget(self.btn_eval)
        self.btn_export_csv = QtWidgets.QPushButton("匯出結果 CSV")
        self.btn_save_heat = QtWidgets.QPushButton("批次另存熱圖")
        ctrl.addWidget(self.btn_export_csv)
        ctrl.addWidget(self.btn_save_heat)
        self.chk_suppress = QtWidgets.QCheckBox("抑制背景/反光（熱圖）")
        self.chk_suppress.setToolTip("減良品平均圖 + 遮掉鋼珠圓形外，讓熱圖聚焦在瑕疵；不影響分數")
        ctrl.addWidget(self.chk_suppress)
        thr_box = QtWidgets.QGroupBox("門檻調整")
        tg = QtWidgets.QHBoxLayout(thr_box)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.lbl_thr_val = QtWidgets.QLabel("—")
        tg.addWidget(self.slider, 1)
        tg.addWidget(self.lbl_thr_val)
        ctrl.addWidget(thr_box)
        root.addWidget(ctrl_box, 0)

        # 狀態列
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.status.addPermanentWidget(self.progress)

        # 事件
        self.btn_open_img.clicked.connect(self.on_open_image)
        self.btn_open_folder.clicked.connect(self.on_open_folder)
        self.btn_eval.clicked.connect(self.on_eval)
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_save_heat.clicked.connect(self.on_save_heatmaps)
        self.list_widget.currentItemChanged.connect(self.on_select)
        self.slider.valueChanged.connect(self.on_threshold_changed)
        self.chk_suppress.toggled.connect(self.on_suppress_toggled)
        self._set_busy(True)

    def on_suppress_toggled(self, _checked):
        cur = self.list_widget.currentItem()
        if cur is not None:
            self.on_select(cur, None)

    def _image_label(self, title, minsize=260):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        lbl = QtWidgets.QLabel("（無影像）")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setMinimumSize(minsize, minsize)
        lbl.setStyleSheet("background:#222;color:#888;")
        lay.addWidget(lbl)
        return box, lbl

    def _gt_path_for(self, img_path):
        """從測試影像路徑推 ground_truth 遮罩路徑（test/<cat>/x.jpg -> ground_truth/<cat>/x.png）。"""
        cat = os.path.basename(os.path.dirname(img_path))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt = os.path.join(os.path.dirname(self.test_dir), "ground_truth", cat, stem + ".png")
        return gt if os.path.exists(gt) else None

    def _init_axes(self):
        self.ax_hist.set_title("分數分布"); self.ax_hist.set_xlabel("anomaly score")
        self.ax_roc.set_title("ROC"); self.ax_roc.set_xlabel("FPR"); self.ax_roc.set_ylabel("TPR")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # ---------------- 載入 ---------------- #
    def _start_load(self):
        if not os.path.exists(os.path.join(self.weights_dir, self.model.suffix + ".pth")):
            QtWidgets.QMessageBox.warning(
                self, "找不到權重",
                "找不到權重檔：\n%s\n\n請先訓練（python main.py --dataset SteelBall ...），"
                "或用『載入權重資料夾』選擇。" % os.path.join(self.weights_dir, self.model.suffix + ".pth"))
        self.status.showMessage("載入模型中… (device=%s)" % DEVICE)
        self.loader = LoadWorker(self.model, self.train_good)
        self.loader.progress.connect(self._on_progress)
        self.loader.done.connect(self._on_loaded)
        self.loader.start()

    def _on_progress(self, i, n, text):
        self.progress.setMaximum(n); self.progress.setValue(i)
        self.status.showMessage("%s … %d/%d" % (text, i, n))

    def _on_loaded(self, ok, err):
        if not ok:
            QtWidgets.QMessageBox.critical(self, "載入失敗",
                "無法載入模型/權重：\n%s\n\n權重路徑：\n%s" % (err, self.weights_dir))
            self.status.showMessage("載入失敗")
            return
        self.threshold = self.model.default_threshold
        self._setup_slider()
        self.status.showMessage("模型就緒（device=%s）。可開啟影像或評估測試集。" % DEVICE)
        self._set_busy(False)
        if os.path.isdir(self.test_dir):
            self._populate_from_test_dir()

    def _setup_slider(self):
        lo = 0.0
        hi = max(self.model.default_threshold * 2.0,
                 float(self.model.train_good_scores.max()) * 2.0)
        self._thr_lo, self._thr_hi = lo, hi
        self.slider.setMinimum(0); self.slider.setMaximum(1000)
        self.slider.setEnabled(True)
        self._set_slider_to(self.threshold)

    def _set_slider_to(self, val):
        frac = (val - self._thr_lo) / (self._thr_hi - self._thr_lo + 1e-9)
        self.slider.blockSignals(True)
        self.slider.setValue(int(np.clip(frac, 0, 1) * 1000))
        self.slider.blockSignals(False)
        self.lbl_thr_val.setText("%.4f" % val)

    # ---------------- 影像清單 ---------------- #
    def _populate_from_test_dir(self):
        self.list_widget.clear()
        for sub in sorted(os.listdir(self.test_dir)):
            d = os.path.join(self.test_dir, sub)
            if not os.path.isdir(d):
                continue
            for f in list_images(d):
                it = QtWidgets.QListWidgetItem("[%s] %s" % (sub, os.path.basename(f)))
                it.setData(QtCore.Qt.UserRole, f)
                self.list_widget.addItem(it)

    def on_open_image(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇影像", self.test_dir, "影像 (*.jpg *.jpeg *.png *.bmp)")
        if f:
            it = QtWidgets.QListWidgetItem(os.path.basename(f))
            it.setData(QtCore.Qt.UserRole, f)
            self.list_widget.addItem(it)
            self.list_widget.setCurrentItem(it)

    def on_open_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇資料夾", self.test_dir)
        if not d:
            return
        self.list_widget.clear()
        for f in list_images(d):
            it = QtWidgets.QListWidgetItem(os.path.basename(f))
            it.setData(QtCore.Qt.UserRole, f)
            self.list_widget.addItem(it)

    # ---------------- 單張推論 ---------------- #
    def on_select(self, cur, _prev):
        if cur is None or not self.model.is_ready():
            return
        path = cur.data(QtCore.Qt.UserRole)
        self.cur_image_path = path
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            self.status.showMessage("無法開啟影像：%s" % e)
            return
        score, disp = self.model.predict(pil, suppress_bg=self.chk_suppress.isChecked())
        self.cur_score = score
        self.cur_disp_map = disp
        self._show_image(pil, disp)
        self._show_pred(pil)
        self._show_mask(path)
        self._update_verdict(score)

    def _show_pred(self, pil):
        pm = self.model.predicted_mask(pil)               # uint8 0/255
        self.lbl_pred[1].setPixmap(self._np_to_pix(np.stack([pm] * 3, -1), self.lbl_pred[1]))

    def _show_mask(self, img_path):
        gt = self._gt_path_for(img_path)
        if gt is None:
            self.lbl_mask[1].setPixmap(QtGui.QPixmap())
            self.lbl_mask[1].setText("（無遮罩 / 良品）")
            return
        m = cv2.imread(gt, 0)
        if m is None:
            self.lbl_mask[1].setText("（遮罩讀取失敗）")
            return
        self.lbl_mask[1].setPixmap(self._np_to_pix(np.stack([m] * 3, -1), self.lbl_mask[1]))

    def _show_image(self, pil, disp_map):
        self.lbl_orig[1].setPixmap(self._pil_to_pix(pil, self.lbl_orig[1]))
        rgb = np.array(pil)[:, :, ::-1].copy()                 # to BGR
        heat = cv2.applyColorMap(np.clip(disp_map, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)
        overlay = overlay[:, :, ::-1].copy()                   # to RGB
        self.lbl_heat[1].setPixmap(self._np_to_pix(overlay, self.lbl_heat[1]))

    def _update_verdict(self, score):
        thr = self.threshold
        is_ng = (thr is not None) and (score > thr)
        self.lbl_verdict.setText("NG（瑕疵）" if is_ng else "OK（良品）")
        self.lbl_verdict.setStyleSheet("color:#e74c3c;" if is_ng else "color:#2ecc71;")
        self.lbl_score.setText("異常分數: %.4f    門檻: %s"
                               % (score, "%.4f" % thr if thr is not None else "—"))

    # ---------------- 門檻滑桿 ---------------- #
    def on_threshold_changed(self, vv):
        self.threshold = self._thr_lo + (vv / 1000.0) * (self._thr_hi - self._thr_lo)
        self.lbl_thr_val.setText("%.4f" % self.threshold)
        if self.cur_score is not None:
            self._update_verdict(self.cur_score)
        if self.eval_scores is not None:
            self._refresh_metrics()
            self._draw_charts()

    # ---------------- 批次評估 ---------------- #
    def on_eval(self):
        if not self.model.is_ready():
            return
        if not os.path.isdir(self.test_dir):
            d = QtWidgets.QFileDialog.getExistingDirectory(
                self, "選擇測試集資料夾（內含 good/ 與瑕疵子資料夾）", BASE)
            if not d:
                return
            self.test_dir = d
        self._set_busy(True)
        self.evw = EvalWorker(self.model, self.test_dir)
        self.evw.progress.connect(self._on_progress)
        self.evw.done.connect(self._on_eval_done)
        self.evw.start()

    def _on_eval_done(self, paths, labels, scores, cats):
        self.eval_paths, self.eval_labels, self.eval_scores = paths, labels, scores
        self.eval_cats = cats
        if HAVE_SKLEARN and len(set(labels.tolist())) == 2:
            fpr, tpr, thr = roc_curve(labels, scores)
            j = np.argmax(tpr - fpr)                            # Youden's J
            self.threshold = float(thr[j])
            self._thr_hi = max(self._thr_hi, float(scores.max()) * 1.2)
            self._set_slider_to(self.threshold)
        self._refresh_metrics()
        self._draw_charts()
        if self.cur_score is not None:
            self._update_verdict(self.cur_score)
        self._set_busy(False)
        self.status.showMessage("測試集評估完成，共 %d 張。" % len(paths))

    # ---------------- 匯出 CSV ---------------- #
    def on_export_csv(self):
        if self.eval_scores is None:
            QtWidgets.QMessageBox.information(
                self, "尚未評估", "請先按「評估測試集」產生分數，再匯出 CSV。")
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "匯出結果 CSV", os.path.join(BASE, "uninet_results.csv"), "CSV (*.csv)")
        if not f:
            return
        thr = self.threshold
        try:
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["filename", "category", "category_name", "label",
                            "anomaly_score", "threshold", "prediction", "correct"])
                for p, lab, sc, cat in zip(self.eval_paths, self.eval_labels,
                                           self.eval_scores, self.eval_cats):
                    is_ng = bool(sc > thr)
                    correct = "Y" if is_ng == (lab == 1) else "N"
                    w.writerow([os.path.basename(p), cat, CODE_NAMES.get(cat, ""),
                                "defect" if lab == 1 else "good",
                                "%.6f" % sc, "%.6f" % thr,
                                "NG" if is_ng else "OK", correct])
            self.status.showMessage("已匯出 CSV：%s（%d 列）" % (f, len(self.eval_paths)))
            QtWidgets.QMessageBox.information(self, "完成", "已匯出：\n%s" % f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "匯出失敗", str(e))

    # ---------------- 批次另存熱圖 ---------------- #
    def on_save_heatmaps(self):
        if not self.model.is_ready():
            return
        items = []
        for i in range(self.list_widget.count()):
            p = self.list_widget.item(i).data(QtCore.Qt.UserRole)
            items.append((p, os.path.basename(os.path.dirname(p))))
        if not items:
            QtWidgets.QMessageBox.information(
                self, "沒有影像", "清單沒有影像。請先「開啟資料夾」或載入測試集。")
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇熱圖輸出資料夾", BASE)
        if not out:
            return
        self._set_busy(True)
        self.shw = SaveHeatmapWorker(self.model, items, out, self.threshold,
                                     suppress_bg=self.chk_suppress.isChecked())
        self.shw.progress.connect(self._on_progress)
        self.shw.done.connect(self._on_save_heat_done)
        self.shw.start()

    def _on_save_heat_done(self, cnt, out_dir):
        self._set_busy(False)
        self.status.showMessage("已另存 %d 張熱圖到：%s" % (cnt, out_dir))
        QtWidgets.QMessageBox.information(
            self, "完成", "已另存 %d 張熱圖到：\n%s" % (cnt, out_dir))

    def _refresh_metrics(self):
        labels, scores = self.eval_labels, self.eval_scores
        thr = self.threshold
        pred = (scores > thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        n_good = int((labels == 0).sum()); n_def = int((labels == 1).sum())
        auc = (roc_auc_score(labels, scores) * 100
               if HAVE_SKLEARN and n_good and n_def else float("nan"))
        recall = tp / (tp + fn) * 100 if (tp + fn) else 0
        overkill = fp / (fp + tn) * 100 if (fp + tn) else 0
        html = (
            "<b>影像級 AUC: %.2f%%</b><br>"
            "良品 %d / 瑕疵 %d，門檻 %.4f<br>"
            "瑕疵抓出率(Recall): %.1f%% (%d/%d)<br>"
            "良品過殺率: %.1f%% (%d/%d)<br>"
            "混淆: TP=%d TN=%d FP=%d FN=%d"
            % (auc, n_good, n_def, thr, recall, tp, tp + fn,
               overkill, fp, fp + tn, tp, tn, fp, fn))
        # 每類瑕疵抓出率（依測試集子資料夾分組）
        if self.eval_cats is not None:
            cats = np.array(self.eval_cats)
            rows = ""
            for cat in sorted(set(cats[labels == 1].tolist())):
                idx = (cats == cat) & (labels == 1)
                ntot = int(idx.sum())
                ndet = int((scores[idx] > thr).sum())
                rate = 100.0 * ndet / ntot if ntot else 0.0
                color = "#2ecc71" if rate >= 99.9 else ("#e67e22" if rate >= 80 else "#e74c3c")
                rows += ("<tr><td>%s %s</td><td align='right'>%d/%d</td>"
                         "<td align='right'><font color='%s'>%.1f%%</font></td></tr>"
                         % (cat, CODE_NAMES.get(cat, ""), ndet, ntot, color, rate))
            html += ("<br><b>每類瑕疵抓出率</b>"
                     "<table cellspacing='4'><tr><td><b>類別</b></td>"
                     "<td><b>抓出</b></td><td><b>率</b></td></tr>%s</table>" % rows)
        self.lbl_metrics.setText(html)

    def _draw_charts(self):
        labels, scores = self.eval_labels, self.eval_scores
        self.ax_hist.clear(); self.ax_roc.clear()
        good = scores[labels == 0]; bad = scores[labels == 1]
        bins = 30
        if len(good):
            self.ax_hist.hist(good, bins=bins, alpha=0.6, label="良品 good", color="#2ecc71")
        if len(bad):
            self.ax_hist.hist(bad, bins=bins, alpha=0.6, label="瑕疵 defect", color="#e74c3c")
        if self.threshold is not None:
            self.ax_hist.axvline(self.threshold, color="k", ls="--", lw=1.2, label="門檻")
        self.ax_hist.set_title("分數分布"); self.ax_hist.set_xlabel("anomaly score")
        self.ax_hist.legend(fontsize=8)
        if HAVE_SKLEARN and len(set(labels.tolist())) == 2:
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            self.ax_roc.plot(fpr, tpr, color="#2980b9", label="AUC=%.3f" % auc)
            self.ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
            self.ax_roc.legend(fontsize=8)
        self.ax_roc.set_title("ROC"); self.ax_roc.set_xlabel("FPR"); self.ax_roc.set_ylabel("TPR")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # ---------------- 工具 ---------------- #
    def _set_busy(self, busy):
        for w in (self.btn_open_img, self.btn_open_folder, self.btn_eval,
                  self.btn_export_csv, self.btn_save_heat, self.chk_suppress, self.list_widget):
            w.setEnabled(not busy)
        self.progress.setVisible(busy)

    @staticmethod
    def _pil_to_pix(pil, label):
        return MainWindow._np_to_pix(np.array(pil), label)

    @staticmethod
    def _np_to_pix(arr, label):
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = np.ascontiguousarray(arr)
        h, w, _ = arr.shape
        qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        return pix.scaled(label.width(), label.height(),
                          QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
