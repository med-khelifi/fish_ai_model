"""
Fish Dataset Augmenter
======================
For each source image generates N augmented versions:
  - 80% : keep original background + flip / rotate / mirror / brightness
  - 20% : remove background + solid or gradient bg + same transforms

Usage:
    pip install opencv-python pillow numpy
    pip install "rembg[cpu]"    <- needed for the 20% bg removal

    python augment_dataset.py --dataset ./dataset --count 200
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

FISH_FOLDERS = [
    "001_الوراطة",    "002_القاروص",    "003_السردينة",  "004_البوري",
    "005_السكمبري",   "006_التريليا",   "007_المرجان",   "008_المنكوس",
    "009_البلاميطة",  "010_البوقا",     "011_الشلبة",    "012_المرلان",
    "013_الغزال",     "014_اللقّوق",    "015_الجغالي",   "016_الشامة",
    "017_أبو_سيف",    "018_التن_الأحمر","019_سمكة_الأرنب","020_سمكة_العقرب",
    "021_سمكة_الأسد", "022_السمكة_الصندوق","023_دجاج_البحر","024_السلطعون_الأزرق",
    "025_الحنش",      "026_الكلب_بحر",  "027_قرش_المطرقة","028_القرنيط",
    "029_الحبار",     "030_الجمبري",    "031_الكركند",   "032_سمكة_الحجر",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Backgrounds (for the 20%) ───────────────────────────────────────────────

def solid_bg(h, w):
    c = np.random.randint(30, 220, 3, dtype=np.uint8)
    return np.full((h, w, 3), c, dtype=np.uint8)

def gradient_bg(h, w):
    c1 = np.random.randint(10, 200, 3).astype(np.float32)
    c2 = np.random.randint(10, 200, 3).astype(np.float32)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        t = i / h
        bg[i] = (1 - t) * c1 + t * c2
    return bg.clip(0, 255).astype(np.uint8)

BG_GEN = [solid_bg, solid_bg, gradient_bg, gradient_bg]  # alternates

# ─── I/O (Unicode-safe for Arabic paths on Windows) ──────────────────────────

def read_img(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def save_img(img, path):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    buf.tofile(str(path))

# ─── Background removal ───────────────────────────────────────────────────────

def mask_threshold(img, thr=240):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask

_rembg_fn = None

def mask_rembg(img):
    global _rembg_fn
    if _rembg_fn is None:
        from rembg import remove as _r
        _rembg_fn = _r
    from PIL import Image
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out  = _rembg_fn(pil)
    rgba = np.array(out)
    fish = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    return rgba[:, :, 3], fish   # (mask, fish_no_bg)

def composite(fish, mask, bg):
    h, w = fish.shape[:2]
    bg   = cv2.resize(bg, (w, h))
    m    = cv2.merge([mask]*3).astype(np.float32) / 255.0
    out  = fish.astype(np.float32) * m + bg.astype(np.float32) * (1 - m)
    return out.clip(0, 255).astype(np.uint8)

# ─── Geometry transforms ──────────────────────────────────────────────────────

ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
FLIPS     = [None, 1, 0, -1]   # none / horizontal / vertical / both

def apply_geometry(img, idx):
    rot  = ROTATIONS[(idx // 4) % 4]
    flip = FLIPS[idx % 4]
    out  = img.copy()
    if rot  is not None: out = cv2.rotate(out, rot)
    if flip is not None: out = cv2.flip(out, flip)
    return out

def apply_brightness(img):
    alpha = random.uniform(0.75, 1.25)
    beta  = random.randint(-30, 30)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# ─── Per-image augmentation ───────────────────────────────────────────────────

def augment_one(img_path: Path, count: int, img_size: int,
                use_rembg: bool, threshold: int):

    img = read_img(img_path)
    if img is None:
        print(f"    [!] Cannot read {img_path.name} — skipping")
        return 0

    out_dir = img_path.parent
    stem    = img_path.stem

    n_nobg = max(1, int(count * 0.20))
    n_bg   = count - n_nobg

    saved = 0

    # ── 80% : original background, just transform ────────────────────────
    for i in range(n_bg):
        out = apply_geometry(img, i)
        out = apply_brightness(out)
        out = cv2.resize(out, (img_size, img_size))
        save_img(out, out_dir / f"{stem}_bg_{i:04d}.jpg")
        saved += 1

    # ── 20% : remove bg → solid or gradient bg → transform ───────────────
    if use_rembg:
        try:
            mask, fish = mask_rembg(img)
        except Exception as e:
            print(f"\n    [!] rembg failed ({e}) — using threshold")
            mask = mask_threshold(img, threshold)
            fish = img
    else:
        mask = mask_threshold(img, threshold)
        fish = img

    for i in range(n_nobg):
        h, w   = fish.shape[:2]
        bg     = BG_GEN[i % len(BG_GEN)](h, w)
        frame  = composite(fish, mask, bg)
        frame  = apply_geometry(frame, i)
        frame  = apply_brightness(frame)
        frame  = cv2.resize(frame, (img_size, img_size))
        save_img(frame, out_dir / f"{stem}_newbg_{i:04d}.jpg")
        saved += 1

    return saved

# ─── Main ────────────────────────────────────────────────────────────────────

def run(dataset_dir, count, img_size, use_rembg, threshold):
    dataset     = Path(dataset_dir)
    total_saved = 0
    total_imgs  = 0

    print(f"\n{'='*55}")
    print(f"  Fish Dataset Augmenter")
    print(f"  Dataset : {dataset.resolve()}")
    print(f"  Per img : {count}  →  {count - max(1,int(count*0.2))} keep bg  +  {max(1,int(count*0.2))} new bg")
    print(f"  rembg   : {'yes' if use_rembg else 'no  (use --rembg for better results)'}")
    print(f"{'='*55}\n")

    for folder_name in FISH_FOLDERS:
        folder = dataset / folder_name
        if not folder.exists():
            print(f"[SKIP] {folder_name}  (not found)")
            continue

        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
        if not images:
            print(f"[SKIP] {folder_name}  (no images)")
            continue

        print(f"[+] {folder_name}  ({len(images)} image(s))")
        for img_path in images:
            print(f"    {img_path.name} ...", end=" ", flush=True)
            n = augment_one(img_path, count, img_size, use_rembg, threshold)
            print(f"{n} saved")
            total_saved += n
            total_imgs  += 1

    print(f"\n{'='*55}")
    print(f"  DONE — {total_saved} images from {total_imgs} source images")
    print(f"{'='*55}\n")

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="./dataset",   help="Path to dataset root folder")
    parser.add_argument("--count",     type=int, default=200, help="Augmented images per source image")
    parser.add_argument("--size",      type=int, default=640, help="Output image size in px")
    parser.add_argument("--rembg",     action="store_true",   help="Use rembg for background removal")
    parser.add_argument("--threshold", type=int, default=240, help="Threshold BG removal (0-255)")
    args = parser.parse_args()

    run(args.dataset, args.count, args.size, args.rembg, args.threshold)