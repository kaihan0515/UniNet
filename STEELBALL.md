# Steel-Ball (鋼珠) Surface-Defect Detection with UniNet

This fork adapts the official **UniNet** (CVPR 2025) anomaly-detection framework to
**steel-ball surface-defect inspection**, as a custom one-class industrial AD dataset
named `SteelBall`.

> Upstream: [pangdatangtt/UniNet](https://github.com/pangdatangtt/UniNet) — *UniNet: A
> Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly
> Detection*, Shun Wei, Jielin Jiang, Xiaolong Xu (CVPR 2025). MIT License.

## Concept

A steel ball is **one object** that can carry **12 defect types**. This maps exactly onto
the MVTec-AD single-category layout (one object = `good` + several defect sub-types). The
model trains **only on OK (good) images** and flags anything that deviates as a defect,
producing both an **image-level** OK/NG score and a **pixel-level** localization map.

### The 12 defect types (folder code → name)

| code | 名稱 | code | 名稱 |
|------|------|------|------|
| 100 | 小黑傷 | 106 | 刮傷 |
| 101 | 灰傷、刻痕 | 107 | 生鏽 |
| 102 | 麻點 | 108 | 霧面 |
| 103 | 大黑傷 | 109 | 亮傷-暗 |
| 104 | 研磨傷 | 110 | 小白點、線 |
| 105 | 肯傷 | 111 | 亮傷-亮 |

## Dataset layout

The data lives **outside** the repo at `../data/steelball/steelball/` (UniNet reads from
`../data/<dataset>/<class>`), in MVTec-AD format:

```
../data/steelball/steelball/
├── train/good/          # 89 OK images (training)
├── test/
│   ├── good/            # 15 held-out OK images
│   ├── 100/ … 111/      # defect images per type
└── ground_truth/
    └── 100/ … 111/      # binary masks (0/255), one per defect image
```

- **OK source:** `../data/OK 2/` (104 unique images, de-duplicated by content; 15 held out for `test/good`).
- **Defect source:** `../data/steel ball dataset/` (each defect `.jpg` has a matching LabelMe `.json`).
- **Masks:** rendered from the LabelMe polygon / rectangle / circle annotations at each image's
  native resolution.

### Rebuild the dataset

```bash
python prepare_data/prepare_steelball.py
```

Edit the paths at the top of `prepare_data/prepare_steelball.py` if your raw data lives elsewhere.

## Environment

The upstream pins `python 3.9.7` + `torch 1.12.0+cu113` (`requirements.txt`). This fork was
verified to also run on `python 3.10` + `torch 2.5.1 (cu121)` — you only additionally need
`scikit-image` and `tabulate`:

```bash
pip install scikit-image tabulate     # if using a newer torch env
```

## Train & evaluate

Run from the repo root (so `../data` and `./ckpts` resolve correctly):

```bash
python main.py --dataset SteelBall --setting oc --epochs 100 --batch_size 8
```

- `--setting oc` — one-class (unsupervised) anomaly detection; forced automatically for `SteelBall`.
- Evaluation runs every 10 epochs and saves the best weights to `ckpts/SteelBall/steelball/`.
- Reports **image AUROC**, **pixel AUROC**, and **pixel AUPRO**.

Test only (after a checkpoint exists):

```bash
python main.py --dataset SteelBall --setting oc --load_ckpts
```

## Results (256px, one-class)

Full 100-epoch training on a healthy GPU reaches (best checkpoint):

| metric | value |
|--------|-------|
| image AUROC | **100.0** |
| pixel AUROC | **~85** |
| pixel AUPRO | **~57** |

The defects are tiny (median ~0.1% of image area), so image-level detection saturates
early while pixel-level localization needs the full run to converge.

## Hardware notes (RTX 2080 Ti, 11 GB)

UniNet uses **two** wide_resnet50_2 backbones (source + target teacher) plus the student
and DFS, so it is memory-heavy:

- `256px @ batch 4` ≈ 7.6 GB — **recommended on 11 GB**.
- `256px @ batch 8` ≈ 10.9 GB — fits when the GPU is otherwise idle.
- `288px` / `384px` — exceed 11 GB (cuDNN/OOM); also note UniNet's evaluation hardcodes
  256px anomaly maps, so non-256 resolutions break pixel evaluation. Stick to 256px here.

The evaluation phase (every 10 epochs) is the memory peak. If a run dies *silently*
during eval (no Python traceback) after the GPU was churned by killed processes, the
driver/CUDA state is likely degraded — **reboot** and rerun; the run completes normally on
a clean GPU. `prepare_data/run_steelball_256.sh` waits for a free GPU and runs one clean
256px train+test; it sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to limit
fragmentation.

## What was changed vs. upstream

- `datasets.py` — registered `SteelBall` in the `industrial` + `unsupervised` lists, added
  `steelball_list`, and a one-class loader branch reusing `MVTecDataset(dataset='steelball')`.
- `main.py` — added `SteelBall` to the CLI choices and dataset routing (forces `oc`).
- `train_unsupervisedAD.py` — included `SteelBall` in the industrial-AD eval/checkpoint-save path.
- `eval.py` — made the AUPRO computation's `multiprocessing.Pool` fall back to a sequential
  pass on Windows `PermissionError`/`OSError`, so evaluation never crashes the run.
- `prepare_data/prepare_steelball.py` — **new** dataset + mask builder (this adaptation).
- `prepare_data/run_steelball_256.sh` — **new** helper: waits for a free GPU, then runs one
  clean 256px train+test (batch 4, expandable-segments allocator).
