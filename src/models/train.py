import os
import csv
 
import torch
from tqdm import tqdm
from src.models.zoo import build_retinanet, build_fcos
from src.utils.helper import split_losses, DetectionEvaluator

 
 
 
def build_model(name, num_classes, pretrained_backbone):
    if name == "retinanet":
        return build_retinanet(num_classes, pretrained_backbone)
    elif name == "fcos":
        return build_fcos(num_classes, pretrained_backbone)
    raise ValueError(f"unknown model '{name}' - expected 'retinanet' or 'fcos'")
 
 
def train_one_epoch(model, loader, optimizer, device, epoch, scaler=None):
    model.train()
    total_loss, total_cls_loss, total_bbox_loss = 0.0, 0.0, 0.0
 
    pbar = tqdm(loader, desc=f"[train] epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
        optimizer.zero_grad()
 
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                cls_loss, bbox_loss = split_losses(loss_dict)
                loss = cls_loss + bbox_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            cls_loss, bbox_loss = split_losses(loss_dict)
            loss = cls_loss + bbox_loss
            loss.backward()
            optimizer.step()
 
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")
 
    n = max(1, len(loader))
    return total_loss / n, total_cls_loss / n, total_bbox_loss / n
 
 
@torch.no_grad()
def validate_loss(model, loader, device, epoch):
    # torchvision detection models only return losses in train() mode -
    # kept in train() here but wrapped in no_grad so nothing gets updated.
    model.train()
    total_loss, total_cls_loss, total_bbox_loss = 0.0, 0.0, 0.0
 
    pbar = tqdm(loader, desc=f"[val-loss] epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
        loss_dict = model(images, targets)
        cls_loss, bbox_loss = split_losses(loss_dict)
        loss = cls_loss + bbox_loss
 
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")
 
    n = max(1, len(loader))
    return total_loss / n, total_cls_loss / n, total_bbox_loss / n
 
 
@torch.no_grad()
def validate_detections(model, loader, device, epoch, iou_threshold, score_threshold):
    model.eval()
    evaluator = DetectionEvaluator(iou_threshold=iou_threshold, score_threshold=score_threshold)

    pbar = tqdm(loader, desc=f"[val-metrics] epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # <-- fixed line

        predictions = model(images)  # list of dicts: boxes, labels, scores
        evaluator.update(predictions, targets)

    return evaluator.compute()
 
 
def save_metrics_row(csv_path, row, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
 