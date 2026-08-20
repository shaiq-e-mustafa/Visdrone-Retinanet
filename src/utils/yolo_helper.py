import shutil
from PIL import Image
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import os

def convert_annotations(ann_path, img_w, img_h, class_map):
    yolo_lines = []

    with open(ann_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            values = line.split(",")
            if len(values) < 6:
                continue

            x, y, w, h = map(int, values[0:4])
            category = int(values[5])

            if category not in class_map:
                continue
            if w <= 0 or h <= 0:
                continue

            new_class = class_map[category]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            yolo_lines.append(f"{new_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return yolo_lines


def process_one(pair, dst_img_dir, dst_lbl_dir, copy_mode, class_map):
    """Worker function - must be top-level (picklable) for multiprocessing."""
    img_path, ann_path = pair
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]

    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size  # reads header only, does not decode pixels
    except Exception as e:
        return ("error", fname, str(e))

    yolo_lines = convert_annotations(ann_path, img_w, img_h, class_map)

    out_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
    with open(out_lbl_path, "w") as f:
        f.write("\n".join(yolo_lines))

    dst_img_path = os.path.join(dst_img_dir, fname)
    if copy_mode == "copy":
        shutil.copy2(img_path, dst_img_path)
    elif copy_mode == "link":
        try:
            os.link(img_path, dst_img_path)  # hardlink, near-instant, same filesystem only
        except OSError:
            shutil.copy2(img_path, dst_img_path)  # fallback (e.g. cross-device)
    # copy_mode == "none" -> skip image entirely (labels only)

    status = "empty" if len(yolo_lines) == 0 else "ok"
    return (status, fname, None)


def process_split(split_name, pairs, dst_root, copy_mode, workers, chunk_size, class_map):
    dst_img_dir = os.path.join(dst_root, split_name, "images")
    dst_lbl_dir = os.path.join(dst_root, split_name, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    worker_fn = partial(
        process_one,
        dst_img_dir=dst_img_dir,
        dst_lbl_dir=dst_lbl_dir,
        copy_mode=copy_mode,
        class_map=class_map
    )

    results = []
    with Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(worker_fn, pairs, chunksize=chunk_size)
        iterator = tqdm(iterator, total=len(pairs), desc=split_name)
        for r in iterator:
            results.append(r)

    n_ok = sum(1 for r in results if r[0] == "ok")
    n_empty = sum(1 for r in results if r[0] == "empty")
    n_error = sum(1 for r in results if r[0] == "error")

    print(f"[{split_name}] {n_ok} converted, {n_empty} with zero boxes (kept as background), {n_error} errors")
    for r in results:
        if r[0] == "error":
            print(f"  [ERROR] {r[1]}: {r[2]}")
