# 鋼珠瑕疵檢測 — UniNet 實作進度報告

> 專案：機器視覺應用（MVA）鋼珠（steel ball）表面瑕疵檢測
> 更新日期：2026-06-20
> 程式碼倉庫（fork）：<https://github.com/kaihan0515/UniNet>（forked from `pangdatangtt/UniNet`）

---

## 1. 摘要

本專案將 **UniNet**（*Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection*，**CVPR 2025**，官方倉庫 `pangdatangtt/UniNet`）適配到**鋼珠表面瑕疵檢測**，並建立兩條互補的模型與一套 PyQt5 圖形介面：

| 模型 | 任務 | 核心結果 |
|------|------|----------|
| **UniNet（異常偵測）** | 判斷「**是否有瑕疵**」(OK/NG) | image AUROC **100**、pixel AUROC **83.2**、pixel AUPRO **58.6** |
| **ResNet18（多類別分類器）** | 判斷「**是哪一種瑕疵**」(13 類) | test accuracy **90.6%** |

兩者搭配：UniNet 負責「抓出不良品」，分類器負責「判斷瑕疵類型」。

---

## 2. 資料集

### 2.1 來源
- **良品（Normal/good）**：`data/OK 2/`，104 張（去重後）。
- **瑕疵**：`data/steel ball dataset/`，12 類、共 961 張，每張附 LabelMe 多邊形標註 JSON。

### 2.2 UniNet（MVTec-AD 格式）
由 `prepare_data/prepare_steelball.py` 建置到 `data/steelball/steelball/`：

```
steelball/
├── train/good/        89 張（OK 影像，僅良品訓練）
├── test/
│   ├── good/          15 張保留良品
│   └── 100…111/       12 類瑕疵影像（共 961 張）
└── ground_truth/
    └── 100…111/       961 張二值遮罩（由 LabelMe JSON 多邊形/矩形/圓形轉出）
```

> 觀念：一顆鋼珠是「一個物件」、有「12 種瑕疵型態」，對應 MVTec 單物件結構。模型**只用良品訓練**，將偏離視為瑕疵。

### 2.3 12 種瑕疵類別

| 代碼 | 名稱 | 代碼 | 名稱 |
|------|------|------|------|
| 100 | 小黑傷 | 106 | 刮傷 |
| 101 | 灰傷、刻痕 | 107 | 生鏽 |
| 102 | 麻點 | 108 | 霧面 |
| 103 | 大黑傷 | 109 | 亮傷-暗 |
| 104 | 研磨傷 | 110 | 小白點、線 |
| 105 | 肯傷 | 111 | 亮傷-亮 |

**特性**：瑕疵極小（多邊形面積中位數約佔影像 0.1%），對像素級定位是挑戰。

---

## 3. UniNet 異常偵測

### 3.1 程式碼適配
在原始碼註冊 `SteelBall` 為一類（one-class）工業異常偵測資料集：

- `datasets.py` — 加入 `industrial`/`unsupervised` 清單、`steelball_list`、一類載入分支（重用 `MVTecDataset`，路徑 `../data/steelball`）。
- `main.py` — CLI 選項與資料路由（強制 `oc`）。
- `train_unsupervisedAD.py` — 納入工業 AD 的評估/存權重流程；並**每個 epoch 寫 `train_log.csv`**（loss/指標，供畫 loss 曲線）。

### 3.2 訓練與結果

```bash
python main.py --dataset SteelBall --setting oc --epochs 100 --batch_size 4 --image_size 256
```

| 指標 | 數值 | 說明 |
|------|------|------|
| **Image AUROC** | **100** | 影像級 OK/NG 完美分離（可上線等級） |
| **Pixel AUROC** | **83.2** | 像素級瑕疵定位（小瑕疵下屬中上） |
| **Pixel AUPRO** | **58.6** | 區域重疊品質 |

> 良品分數約 0.58、瑕疵約 1.9–2.7，分離乾淨（對應 AUROC 100）。結果可重現（seed 1203）。

### 3.3 環境與硬體限制
- 環境：conda `MVA_py310_cu121`（torch 2.5.1 + CUDA 12.1，RTX 2080 Ti **11 GB**）。
- **僅支援 256px**：UniNet 同時使用兩個 wide_resnet50_2（source/target teacher）+ student + DFS，記憶體吃重。
  - `256px @ batch 4 ≈ 7.6 GB`（建議）。
  - `384px` 超過 11 GB（cuDNN/OOM）；且 UniNet 評估**寫死 256px 異常圖**，非 256 會壞掉。

### 3.4 Windows 穩定性修正
- `eval.py` 的 AUPRO 原用 `multiprocessing.Pool(8)`，在此機器會 `WinError 5` 崩潰 → 改為 **ThreadPool**（多執行緒，無子進程，快且穩定）。
- ⚠️ **操作守則**：訓練中途要停請用 `Ctrl+C`；**勿用工作管理員強制 kill CUDA 程式**（會使顯卡驅動進入不穩定狀態，需重開機）。

---

## 4. 多類別瑕疵分類器

UniNet 只輸出 OK/NG，**無法**產生「類別 × 類別」混淆矩陣。故另訓練監督式分類器：

```bash
python train_classifier.py --epochs 40 --batch 32 --img_size 224
```

- 模型：ResNet18（ImageNet 預訓練）遷移學習，13 類 = Normal + 12 瑕疵。
- 資料：Normal(OK 2) + 12 類瑕疵，分層 80/20（train 852 / test 213），類別權重處理不平衡 + 資料增強。
- **結果：test accuracy 90.6%**。多數類別 0.95–1.00；弱類為**灰傷刻痕（0.27）、小白點線（0.20）**，主因測試樣本太少（5–15 張）。
- 輸出：`figs/classifier_confusion_matrix.png`（標準正規化混淆矩陣）、`classifier_loss_curve.png`、`classifier_report.txt`、`ckpts/classifier/steelball_resnet18.pth`。

---

## 5. 視覺化與圖形介面

### 5.1 異常熱圖視覺化 `visualize_steelball.py`
批次輸出 `[原圖+GT | 異常熱圖 | GT 遮罩]` 三聯圖（全域統一色階）。
- `--suppress`：**抑制背景/反光**（見 §6）。

### 5.2 PyQt5 MVC 介面 `uninet_gui.py`

```bash
python uninet_gui.py   # 從 repo 根目錄、環境 MVA_py310_cu121
```

| 層 | 內容 |
|----|------|
| **Model** | `UniNetADModel`：載入權重、`predict→(分數, 熱圖)`、良品分數→預設門檻、良品平均圖（背景基準） |
| **View** | 左：三聯影像（原圖／熱圖疊加／**GT 遮罩**）+ OK/NG 判定；中：分數分布 + ROC；右：**操作區** |
| **Controller** | slot + 背景執行緒（載入、批次評估、批次出圖） |

**功能**：單張/資料夾推論、OK/NG 判定、門檻滑桿、測試集評估（AUC/ROC、**每類瑕疵抓出率**、過殺率/抓出率/混淆數）、**匯出結果 CSV**、**批次另存熱圖**、**抑制背景/反光**勾選框。熱圖正規化為 **0–255**。所有可點選控制集中於右側操作區。

---

## 6. 熱圖「吃到背景/反光」處理

**問題**：異常熱圖會對鋼珠的**反光環狀結構**與角落背景誤判（小瑕疵時尤其明顯），這也是 pixel AUROC 卡在 ~83 的主因。

**後處理（GUI 勾選框 / `--suppress`）**：
1. **減良品平均異常圖**（載入時用 train/good 建立基準）→ 削弱每顆球都有的反光環。
2. **遮掉 Hough 偵測到的鋼珠圓形外** → 乾淨去除角落背景。
3. **異常分數不變**（只清理熱圖，OK/NG 判定不受影響）。

**效果**：角落背景乾淨去除；反光環部分削弱。**根本解法 = 加更多良品重訓**（模型看過足夠多良品後會把反光環學成「正常」）。

---

## 7. 檔案清單（本專案新增/修改）

| 檔案 | 說明 |
|------|------|
| `prepare_data/prepare_steelball.py` | 建 MVTec 格式資料集 + 由 JSON 轉 GT 遮罩 |
| `datasets.py` / `main.py` / `train_unsupervisedAD.py` | 註冊 SteelBall + 每 epoch loss CSV |
| `eval.py` | AUPRO 改 ThreadPool（Windows 穩定） |
| `prepare_data/run_steelball_256.sh` | 等 GPU 空閒後乾淨訓練 256px |
| `visualize_steelball.py` | 異常熱圖批次出圖（全域色階、`--suppress`） |
| `uninet_gui.py` | PyQt5 MVC 成效檢視介面 |
| `train_classifier.py` | 多類別分類器 + N×N 混淆矩陣 + loss 曲線 |
| `plot_training.py` | UniNet loss 曲線 + 每類別（×OK/NG）混淆矩陣 |
| `STEELBALL.md` | 使用說明 |

---

## 8. 待辦 / 未來改進

| 優先 | 項目 | 說明 |
|------|------|------|
| ⭐ 高 | **加更多良品重訓 UniNet** | 目前僅 89 張良品；`SB images` 另有 ~1987 張。最能提升 pixel 定位、並從根本消除反光環誤判 |
| ⭐ 高 | **UniNet loss 曲線** | 現有模型訓練前未存 log；用新版程式重訓一次即自動產生 `train_log.csv`，再 `python plot_training.py` |
| 中 | 分類器弱類改善 | 灰傷刻痕/小白點線蒐集更多影像、或換 ResNet50 |
| 中 | 調 UniNet 超參數 | `alpha/beta/T` 為 MVTec 預設，可針對鋼珠微調 |
| 低 | 部署門檻與報表 | 以 Youden's J 定門檻，輸出 precision/recall/F1 與混淆矩陣 |

---

## 9. 快速指令彙整

```bash
conda activate MVA_py310_cu121
cd D:\111370211\MVA\final\UniNet

# 1) 建資料集（含遮罩）
python prepare_data/prepare_steelball.py

# 2) 訓練 UniNet 異常偵測（會自動存 train_log.csv + 權重）
python main.py --dataset SteelBall --setting oc --epochs 100 --batch_size 4 --image_size 256

# 3) 異常熱圖（可加 --suppress 抑制背景）
python visualize_steelball.py --per_cat 4 --good 4 --suppress

# 4) 成效檢視 GUI
python uninet_gui.py

# 5) 多類別分類器（→ N×N 混淆矩陣 + loss 曲線）
python train_classifier.py --epochs 40

# 6) UniNet loss 曲線 + 每類別混淆矩陣（需先有 train_log.csv / GUI 匯出的 CSV）
python plot_training.py
```
