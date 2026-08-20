from glob import glob
import os


def list_stems(images_dir, annotations_dir):
    print("images dir", images_dir)
    print("ann dir", annotations_dir)
    image_files = [
        f for f in glob(os.path.join(images_dir, "*.jpg"))
    ]
    if len(image_files) == 0:
        raise ValueError("No images found")
    pairs = []
    missing = 0
    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(annotations_dir, stem + ".txt")
        if os.path.exists(ann_path):
            pairs.append((img_path, ann_path))
        else:
            missing += 1

    if missing:
        print(f"[WARN] {missing} images had no matching annotation file and were excluded entirely")

    return pairs

