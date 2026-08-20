"""
Reads RAW VisDrone annotations directly:
    <x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>
(absolute pixel, top-left x/y + width/height)

Applies class_mapper.py's person/vehicle collapse, then offsets by +1 so
label 0 is reserved for background (required by torchvision detection models
- RetinaNet, FCOS, etc. all use this convention internally):

    class_mapper output 0 (person)  -> 1
    class_mapper output 1 (vehicle) -> 2

Category 0 (VisDrone "ignored regions") and any category not present in
class_mapper's dict (e.g. "others"=11, if it shows up in your files) are
skipped rather than raising an error.
"""

import os

import cv2
import torch
from torch.utils.data import Dataset

from src.utils.class_mapper import class_mapper  # adjust import path to wherever class_mapper.py lives in your repo


TORCHVISION_OFFSET = 1  # shift so 0 is reserved for background
NUM_CLASSES_INCL_BACKGROUND = 3  # background + person + vehicle


class RawVisDroneDataset(Dataset):
    def __init__(self, samples, transforms=None):
        """
        samples: list of (image_path, annotation_path) tuples.
        Use get_img_ann.list_stems() + split_pairs.split_pairs() to build
        these (see dataloader.py) rather than constructing this directly.
        """
        self.samples = samples
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        boxes = []
        labels = []

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

                if category not in class_mapper:
                    continue
                if w <= 0 or h <= 0:
                    continue

                # clip to image bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                if x2 <= x1 or y2 <= y1:
                    continue

                new_class = class_mapper[category] + TORCHVISION_OFFSET
                boxes.append([x1, y1, x2, y2])
                labels.append(new_class)

        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if len(boxes) else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if len(labels) else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)