# src/utils/tile_dataset.py
import os
from glob import glob
import random
import cv2


def compute_tiles(img_w, img_h, tile_size, overlap):
    stride = int(tile_size * (1 - overlap))
    xs = list(range(0, max(img_w - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(img_h - tile_size, 0) + 1, stride)) or [0]
    if xs[-1] + tile_size < img_w:
        xs.append(max(img_w - tile_size, 0))
    if ys[-1] + tile_size < img_h:
        ys.append(max(img_h - tile_size, 0))
    xs, ys = sorted(set(xs)), sorted(set(ys))
    return [(x0, y0, min(x0 + tile_size, img_w), min(y0 + tile_size, img_h))
            for y0 in ys for x0 in xs]


def clip_boxes_to_tile(lines, tile, min_visibility=0.3):
    x0, y0, x1, y1 = tile
    kept = []
    for line in lines:
        vals = line.strip().split(",")
        if len(vals) < 8:
            continue
        bx, by, bw, bh = map(int, vals[0:4])
        rest = vals[4:]
        bx2, by2 = bx + bw, by + bh
        orig_area = bw * bh
        if orig_area <= 0:
            continue

        ix0, iy0 = max(bx, x0), max(by, y0)
        ix1, iy1 = min(bx2, x1), min(by2, y1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        if iw * ih <= 0 or (iw * ih) / orig_area < min_visibility:
            continue  # box mostly outside this tile - drop, keep in whichever tile does contain it fully

        kept.append(",".join([str(ix0 - x0), str(iy0 - y0), str(iw), str(ih)] + rest))
    return kept


def tile_dataset(images_dir, ann_dir, out_images_dir, out_ann_dir,
                  tile_size=768, overlap=0.25, min_visibility=0.3,
                  keep_empty_frac=0.1, seed=42):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    rng = random.Random(seed)

    for img_path in glob(os.path.join(images_dir, "*.jpg")):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ann_dir, stem + ".txt")
        if not os.path.exists(ann_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(ann_path) as f:
            lines = f.readlines()

        for i, tile in enumerate(compute_tiles(w, h, tile_size, overlap)):
            x0, y0, x1, y1 = tile
            tile_lines = clip_boxes_to_tile(lines, tile, min_visibility)

            # keep a small fraction of empty tiles as negatives - dropping ALL
            # empty tiles biases the model toward always predicting something
            if not tile_lines and rng.random() > keep_empty_frac:
                continue

            tile_stem = f"{stem}_tile{i:03d}"
            cv2.imwrite(os.path.join(out_images_dir, tile_stem + ".jpg"), img[y0:y1, x0:x1])
            with open(os.path.join(out_ann_dir, tile_stem + ".txt"), "w") as f:
                f.write("\n".join(tile_lines) + ("\n" if tile_lines else ""))

    print("Tiling complete ->", out_images_dir)


if __name__ == "__main__":
    tile_dataset(
        images_dir="dataset/raw/images",
        ann_dir="dataset/raw/annotations",
        out_images_dir="dataset/tiled/images",
        out_ann_dir="dataset/tiled/annotations",
        tile_size=768, overlap=0.25,
    )