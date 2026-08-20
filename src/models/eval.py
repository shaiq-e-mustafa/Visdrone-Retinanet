"""
Loads an already-trained checkpoint (best.pt) and re-runs the detection
evaluator at several score thresholds, WITHOUT any retraining.

If person recall jumps a lot at a lower threshold (e.g. 0.3) vs. what you
saw at 0.5 during training, the model IS finding people - it's just scoring
them under 0.5 confidence, and this is a threshold/calibration problem you
can fix at inference time (just use a lower threshold for person specifically).

If recall barely moves across thresholds, the model genuinely isn't finding
those people at all, regardless of confidence - that's a training-time
problem (focal loss alpha, oversampling, etc.), not a threshold problem.

Usage:
    python eval_thresholds.py
(reads the same config/config.yaml as your main training run - no CLI args)
"""


import torch

from src.models.train import build_model
from src.utils.helper import DetectionEvaluator


def evaluate_checkpoint_at_thresholds(model, val_loader, device, thresholds, iou_threshold=0.5):
    model.eval()

    # one evaluator per threshold, all fed from the SAME forward passes -
    # avoids re-running the model once per threshold, which would be
    # wasteful given the model output doesn't change, only the score cutoff does
    evaluators = {t: DetectionEvaluator(iou_threshold=iou_threshold, score_threshold=t) for t in thresholds}

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for t, evaluator in evaluators.items():
                evaluator.update(predictions, targets)

    return {t: evaluator.compute() for t, evaluator in evaluators.items()}

