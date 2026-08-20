"""
Computes per-class (person vs vehicle) box width/height statistics directly
from raw VisDrone annotations, in absolute pixels - NOT resized/normalized.

This matters because RetinaNet's anchor generator is defined in absolute
pixel scale at each FPN level. If most person boxes are, say, 15-40px wide
while the smallest default anchor is 32px, there's a real scale mismatch the
model has to fight regardless of how much data or how many epochs you throw
at it - this script gives you the actual numbers instead of a guess.

Usage:
    python box_stats.py
(reads dataset/raw/images + dataset/raw/annotations directly - adjust paths
below if yours differ)
"""

import os
from glob import glob

import numpy as np

from class_mapper import class_mapper as CLASS_MAP

CLASS_NAMES = {0: "person", 1: "vehicle"}

IMAGES_DIR = os.path.join("dataset", "raw", "images")
ANNOTATIONS_DIR = os.path.join("dataset", "raw", "annotations")


def collect_box_sizes():
    ann_files = glob(os.path.join(ANNOTATIONS_DIR, "*.txt"))
    if not ann_files:
        raise ValueError(f"No annotation files found in {ANNOTATIONS_DIR}")

    widths = {0: [], 1: []}
    heights = {0: [], 1: []}

    for ann_path in ann_files:
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = line.split(",")
                if len(values) < 6:
                    continue

                w, h = int(values[2]), int(values[3])
                category = int(values[5])

                if category not in CLASS_MAP:
                    continue
                if w <= 0 or h <= 0:
                    continue

                cls = CLASS_MAP[category]
                widths[cls].append(w)
                heights[cls].append(h)

    return widths, heights


def summarize(name, values):
    arr = np.array(values)
    percentiles = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"  {name:<8} n={len(arr):<7} min={arr.min():<5} "
          f"p5={percentiles[0]:<6.1f} p25={percentiles[1]:<6.1f} "
          f"median={percentiles[2]:<6.1f} p75={percentiles[3]:<6.1f} "
          f"p95={percentiles[4]:<6.1f} max={arr.max():<6}")


if __name__ == "__main__":
    widths, heights = collect_box_sizes()

    for cls, cls_name in CLASS_NAMES.items():
        print(f"\n{cls_name.upper()} (n={len(widths[cls])} boxes):")
        summarize("width", widths[cls])
        summarize("height", heights[cls])

        w_arr = np.array(widths[cls])
        h_arr = np.array(heights[cls])
        # "effective size" = sqrt(w*h), the standard way to collapse box
        # dimensions to a single scale number for anchor-size comparisons
        effective_size = np.sqrt(w_arr * h_arr)
        aspect_ratio = h_arr / w_arr  # >1 = taller than wide

        print(f"  effective_size (sqrt(w*h)): "
              f"p5={np.percentile(effective_size, 5):.1f} "
              f"median={np.percentile(effective_size, 50):.1f} "
              f"p95={np.percentile(effective_size, 95):.1f}")
        print(f"  aspect_ratio (h/w): "
              f"p5={np.percentile(aspect_ratio, 5):.2f} "
              f"median={np.percentile(aspect_ratio, 50):.2f} "
              f"p95={np.percentile(aspect_ratio, 95):.2f}")

    print("\nFor reference, RetinaNet's default FPN anchor sizes start at 32px "
          "(smallest level) up to ~812px (largest level), with aspect ratios "
          "(0.5, 1.0, 2.0) at every level.")