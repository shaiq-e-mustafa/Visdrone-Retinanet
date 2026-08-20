"""
Two things:
  1. split_losses(): pulls apart a torchvision detection loss dict into
     a classification loss and a bounding-box loss, generically - RetinaNet
     returns {'classification', 'bbox_regression'}; FCOS returns
     {'classification', 'bbox_regression', 'bbox_ctrness'}. Everything that
     isn't 'classification' gets summed into "bbox_loss" so this works for
     either model without special-casing.

  2. evaluate_detections(): greedy IoU-based matching to get per-class
     precision/recall from a val pass. This is NOT full COCO mAP (no
     multi-threshold averaging) - it's a fixed IoU + score threshold
     precision/recall, same spirit as the confusion-matrix analysis you were
     already doing on the YOLO runs, so the numbers are comparable to that.
"""

from collections import defaultdict

import torch
from torchvision.ops import box_iou

LABEL_NAMES = {1: "person", 2: "vehicle"}  # matches raw_dataset.py's +1 offset


def split_losses(loss_dict):
    cls_loss = loss_dict.get("classification", torch.tensor(0.0))
    bbox_loss = sum(v for k, v in loss_dict.items() if k != "classification")
    return cls_loss, bbox_loss


class DetectionEvaluator:
    """
    Accumulate TP/FP/FN per class across a full val pass, then call
    .compute() once at the end of the epoch.
    """

    def __init__(self, iou_threshold=0.5, score_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def update(self, predictions, targets):
        """
        predictions: list of dicts (model output in eval() mode), each with
                     'boxes', 'labels', 'scores'
        targets: list of dicts (ground truth), each with 'boxes', 'labels'
        """
        for pred, target in zip(predictions, targets):
            keep = pred["scores"] >= self.score_threshold
            pred_boxes = pred["boxes"][keep]
            pred_labels = pred["labels"][keep]
            pred_scores = pred["scores"][keep]

            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            for cls in LABEL_NAMES:
                cls_pred_mask = pred_labels == cls
                cls_gt_mask = gt_labels == cls

                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]
                cls_gt_boxes = gt_boxes[cls_gt_mask]

                n_gt = cls_gt_boxes.shape[0]
                n_pred = cls_pred_boxes.shape[0]

                if n_gt == 0 and n_pred == 0:
                    continue
                if n_gt == 0:
                    self.fp[cls] += n_pred
                    continue
                if n_pred == 0:
                    self.fn[cls] += n_gt
                    continue

                # sort predictions by score desc for greedy matching
                order = torch.argsort(cls_pred_scores, descending=True)
                cls_pred_boxes = cls_pred_boxes[order]

                ious = box_iou(cls_pred_boxes, cls_gt_boxes)  # [n_pred, n_gt]
                matched_gt = torch.zeros(n_gt, dtype=torch.bool)

                for p_idx in range(n_pred):
                    best_iou, best_gt_idx = ious[p_idx].max(dim=0)
                    if best_iou >= self.iou_threshold and not matched_gt[best_gt_idx]:
                        matched_gt[best_gt_idx] = True
                        self.tp[cls] += 1
                    else:
                        self.fp[cls] += 1

                self.fn[cls] += (~matched_gt).sum().item()

    def compute(self):
        """Returns dict: {class_name: {'precision': ..., 'recall': ...}}"""
        results = {}
        for cls, name in LABEL_NAMES.items():
            tp, fp, fn = self.tp[cls], self.fp[cls], self.fn[cls]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results[name] = {"precision": precision, "recall": recall}
        return results