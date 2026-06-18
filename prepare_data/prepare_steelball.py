# -*- coding: utf-8 -*-
"""Build a UniNet / MVTec-AD format dataset for steel-ball (鋼珠) surface-defect detection.

One object category ("steelball") with 12 defect *types* (numeric codes 100-111),
mirroring the MVTec single-category layout (e.g. bottle -> good + broken_large + ...).

Output (consumed by UniNet's one-class MVTecDataset, which reads from ../data/<dataset>):

    data/steelball/steelball/
        train/good/*.jpg                 <- OK images (from "OK 2"), minus held-out test
        test/good/*.jpg                  <- held-out OK images
        test/<code>/*.jpg                <- defect images per type (100..111)
        ground_truth/<code>/*.png        <- binary masks (0/255) from LabelMe JSON polygons

Run:  python prepare_data/prepare_steelball.py
Originals are never modified; images are copied and masks are rendered fresh.
"""
import os, glob, json, shutil, random, hashlib
import numpy as np
import cv2
from PIL import Image

# --- paths (project layout: repo at .../final/UniNet, data at .../final/data) ---
ROOT = r"D:/111370211/MVA/final/data"
OK_DIR  = os.path.join(ROOT, "OK 2")                 # OK / good source (train + test/good)
DEF_DIR = os.path.join(ROOT, "steel ball dataset")   # 12 defect categories + good, each img has a .json
OUT     = os.path.join(ROOT, "steelball", "steelball")  # -> ../data/steelball/steelball  (UniNet class="steelball")

N_TEST_GOOD = 15          # held-out OK images for test/good
SEED = 42
random.seed(SEED)

# code -> Chinese name (folders are named by ASCII code for path-safety; mapping kept for reference)
CODE_NAMES = {
    "100": "小黑傷", "101": "灰傷、刻痕", "102": "麻點", "103": "大黑傷",
    "104": "研磨傷", "105": "肯傷", "106": "刮傷", "107": "生鏽",
    "108": "霧面", "109": "亮傷-暗", "110": "小白點、線", "111": "亮傷-亮",
}

def reset(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()

def cat_code(folder_name):
    """numeric prefix of a category folder, e.g. '102麻點' -> '102'."""
    digits = "".join(ch for ch in folder_name if ch.isdigit())
    return digits[:3] if digits else folder_name[:3]

def render_mask(json_path, h, w):
    """Render a binary uint8 mask (0/255) from a LabelMe JSON onto an (h, w) canvas."""
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    for s in d.get("shapes", []):
        st = s.get("shape_type", "polygon")
        pts = np.array(s.get("points", []), dtype=np.float32)
        if len(pts) == 0:
            continue
        if st == "polygon" or st == "linestrip":
            poly = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
        elif st == "rectangle":
            (x1, y1), (x2, y2) = pts[0], pts[1]
            cv2.rectangle(mask, (int(round(x1)), int(round(y1))),
                          (int(round(x2)), int(round(y2))), 255, thickness=-1)
        elif st == "circle":
            (cx, cy), (px, py) = pts[0], pts[1]
            r = int(round(float(np.hypot(px - cx, py - cy))))
            cv2.circle(mask, (int(round(cx)), int(round(cy))), max(r, 1), 255, thickness=-1)
        elif st == "line":
            p = np.round(pts).astype(np.int32)
            cv2.polylines(mask, [p], False, 255, thickness=3)
        elif st == "point":
            for px, py in pts:
                cv2.circle(mask, (int(round(px)), int(round(py))), 3, 255, -1)
        else:  # unknown -> treat as polygon
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask

# ---------------- good (OK) split ----------------
ok_imgs = []
for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    ok_imgs += glob.glob(os.path.join(OK_DIR, e))
# de-duplicate by content so an image can't land in both train and test
seen, uniq = set(), []
for f in sorted(ok_imgs):
    h = md5(f)
    if h not in seen:
        seen.add(h); uniq.append(f)
ok_imgs = uniq
random.shuffle(ok_imgs)
test_good, train_good = ok_imgs[:N_TEST_GOOD], ok_imgs[N_TEST_GOOD:]

reset(OUT)                                 # clean slate for the whole dataset
train_good_dir = os.path.join(OUT, "train", "good")
test_good_dir  = os.path.join(OUT, "test", "good")
gt_root        = os.path.join(OUT, "ground_truth")
for d in (train_good_dir, test_good_dir, gt_root):
    os.makedirs(d, exist_ok=True)

for f in train_good:
    shutil.copy2(f, os.path.join(train_good_dir, os.path.basename(f)))
for f in test_good:
    shutil.copy2(f, os.path.join(test_good_dir, os.path.basename(f)))
print(f"train/good : {len(train_good)}")
print(f"test/good  : {len(test_good)}")

# ---------------- defect types -> test/<code> + ground_truth/<code> ----------------
cat_dirs = [d for d in sorted(os.listdir(DEF_DIR))
            if os.path.isdir(os.path.join(DEF_DIR, d)) and d != "good"]

total_def, total_masked, missing_json = 0, 0, 0
for cat in cat_dirs:
    code = cat_code(cat)
    src = os.path.join(DEF_DIR, cat)
    test_cat_dir = os.path.join(OUT, "test", code)
    gt_cat_dir   = os.path.join(gt_root, code)
    reset(test_cat_dir); reset(gt_cat_dir)

    imgs = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.jpeg")))
    n = 0
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        jf = os.path.join(src, stem + ".json")
        # image resolution (trust the file, not just the json)
        with Image.open(img) as im:
            w, h = im.size
        if os.path.exists(jf):
            mask = render_mask(jf, h, w)
            total_masked += 1
        else:
            mask = np.zeros((h, w), dtype=np.uint8)  # keep pair count consistent
            missing_json += 1
        shutil.copy2(img, os.path.join(test_cat_dir, stem + ".jpg"))
        cv2.imwrite(os.path.join(gt_cat_dir, stem + ".png"), mask)
        n += 1
    total_def += n
    print(f"test/{code} ({CODE_NAMES.get(code,'?')}): {n} imgs + masks")

print("-" * 50)
print(f"defect categories : {len(cat_dirs)}")
print(f"test/defect total : {total_def}  (masked={total_masked}, missing_json={missing_json})")
print(f"output            : {OUT}")
