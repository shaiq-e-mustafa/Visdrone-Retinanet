import os
import sys

from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config_loader import get_config
from src.models.yolov8 import get_dataset_yolov8, train_yolo
from src.datasets.rawVisDroneDataLoader import get_dataloaders
from src.models.eval import evaluate_checkpoint_at_thresholds
from src.datasets.transforms import train_transform, val_transform
import torch
from src.models.train import build_model, train_one_epoch, validate_detections, validate_loss, save_metrics_row

def run_yolo(config, train_config):

    """
    Main Runner for file, depends on config path
    """
    split_yolo = config["settings"]["split_yolo"]

    if split_yolo == True:
        CLASS_MAP = {
            1: 0,  # pedestrian -> person
            2: 1,  # person     -> person
            3: 2,  # bicycle    -> vehicle
            4: 3,  # car        -> vehicle
            5: 4,  # van        -> vehicle
            6: 5,  # truck      -> vehicle
            7: 6,  # tuktuk     -> vehicle
            8: 7,  # rickshaw   -> vehicle
            9: 8,  # bus        -> vehicle
            10: 9,  # bike       -> vehicle
        }
        root_dir = config["paths"]["dataset_dir"]
        raw_dir = config["paths"]["raw_dir"]
        images_dir = config["paths"]["images"]
        ann_dir = config["paths"]["annotations"]
        destination_dir = config["paths"]["destination"]

        chunk_size = config["settings"]["chunk_size"]
        train_ratio = config["settings"]["train_ratio"]
        val_ratio = config["settings"]["val_ratio"]
        test_ratio = config["settings"]["test_ratio"]

        workers = config["runner"]["workers"]
        image_mode = config["runner"]["mode"]
        seed = config["runner"]["seed"]

        dataset_dir = os.path.join(root_dir, raw_dir)


        get_dataset_yolov8(
            root_dir=dataset_dir,
            img_dir=images_dir,
            ann_dir=ann_dir,
            train_split=train_ratio,
            val_split=val_ratio,
            test_split=test_ratio,
            workers=workers,
            seed=seed,
            mode=image_mode,
            chunk_size=chunk_size,
            dest_dir=destination_dir,
            class_map = CLASS_MAP
            )

    model = train_config["yolo"]["v8"] 
    epochs = train_config["yolo"]["epochs"] 
    image_size = train_config["yolo"]["image_size"] 
    save_dir = train_config["yolo"]["save_dir"]
    batch_size = train_config["yolo"]["batch_size"]
    yolo_config = os.path.join("config", "yolov8.yaml")
    train_yolo(
        yolo_model=model,
        yolo_config=yolo_config,
        epochs=epochs,
        image_size=image_size,
        save_dir=save_dir,
        batch_size=batch_size,
        run_name='yolov8n_run2'
    )
    
def run_model(config):
    save_dir = config["paths"]["save_dir"]
    run_name = config["paths"]["run_name"]
    
    images_dir = os.path.join("dataset", "raw", config["paths"]["images"])
    ann_dir = os.path.join("dataset", "raw", config["paths"]["annotations"])
    train_ratio = config["settings"]["train_ratio"]
    val_ratio = config["settings"]["val_ratio"]
    test_ratio = config["settings"]["test_ratio"]
    amp = config["settings"]["amp"]
    epochs = config["settings"]["epochs"] 
    lr = config["settings"]["learning_rate"] 
    model_name = config["settings"]["model"] 
    results_csv = config["settings"]["results_csv"]


    NUM_CLASSES_INCL_BACKGROUND = config["settings"]["num_classes"]

    save_dir = os.path.join(save_dir, run_name)
    results_csv = os.path.join(save_dir, results_csv)


    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"""Running data loader with
        images_dir: {images_dir}
        ann_dir: {ann_dir}
        train_ratio: {train_ratio}
        val_ratio: {val_ratio}
        test_ratio: {test_ratio}
          """)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        images_dir=images_dir,
        annotations_dir=ann_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=1
    )

    model = build_model(model_name, NUM_CLASSES_INCL_BACKGROUND, True).to(device)
 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None
 
    best_val_loss = float("inf")
    fieldnames = [
        "epoch",
        "train_loss", "train_cls_loss", "train_bbox_loss",
        "val_loss", "val_cls_loss", "val_bbox_loss",
        "person_precision", "person_recall",
        "vehicle_precision", "vehicle_recall",
    ]

    for epoch in tqdm(range(epochs), desc="epochs"):
        train_loss, train_cls_loss, train_bbox_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler
        )
        val_loss, val_cls_loss, val_bbox_loss = validate_loss(model, val_loader, device, epoch)
        per_class = validate_detections(
            model, val_loader, device, epoch,
            iou_threshold=0.3,
            score_threshold=0.3,
        )
        lr_scheduler.step()
 
        row = {
            "epoch": epoch,
            "train_loss": train_loss, "train_cls_loss": train_cls_loss, "train_bbox_loss": train_bbox_loss,
            "val_loss": val_loss, "val_cls_loss": val_cls_loss, "val_bbox_loss": val_bbox_loss,
            "person_precision": per_class["person"]["precision"],
            "person_recall": per_class["person"]["recall"],
            "vehicle_precision": per_class["vehicle"]["precision"],
            "vehicle_recall": per_class["vehicle"]["recall"],
        }
        save_metrics_row(results_csv, row, fieldnames)
 
        print(
            f"epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"person P/R {per_class['person']['precision']:.3f}/{per_class['person']['recall']:.3f} | "
            f"vehicle P/R {per_class['vehicle']['precision']:.3f}/{per_class['vehicle']['recall']:.3f}"
        )
 
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pt"))
            print(f"  new best val_loss {val_loss:.4f}")
 
    print("Training complete. Metrics saved to", save_dir)


def get_eval():
    save_dir = config["paths"]["save_dir"]
    run_name = config["paths"]["run_name"]
    images_dir = os.path.join("dataset", "raw", config["paths"]["images"])
    ann_dir = os.path.join("dataset", "raw", config["paths"]["annotations"])
    model_name = config["settings"]["model"] 
    checkpoint_path = os.path.join(save_dir, run_name, "retinanet_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = get_dataloaders(
        images_dir=images_dir,
        annotations_dir=ann_dir,
        train_transform=val_transform,  # no augmentation needed for eval
        val_transform=val_transform,
        train_ratio=config["settings"]["train_ratio"],
        val_ratio=config["settings"]["val_ratio"],
        test_ratio=config["settings"]["test_ratio"],
    )

    model = build_model(model_name, 3, pretrained_backbone=True).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    thresholds = [0.2, 0.3, 0.4, 0.5]
    results = evaluate_checkpoint_at_thresholds(model, val_loader, device, thresholds)

    print(f"\n{'Threshold':<12}{'Person P':<12}{'Person R':<12}{'Vehicle P':<12}{'Vehicle R':<12}")
    for t in thresholds:
        p = results[t]["person"]
        v = results[t]["vehicle"]
        print(f"{t:<12}{p['precision']:<12.3f}{p['recall']:<12.3f}{v['precision']:<12.3f}{v['recall']:<12.3f}")

if __name__ == "__main__":
    ## Verify CUDA
    
    import sys
    from pathlib import Path

    ROOT = Path.cwd().parent

    sys.path.append(str(ROOT))
    config = get_config(os.path.join("config", "config.yaml"))
    # train_config = get_config(os.path.join("config", "train_yolo.yaml"))
    # run(config, train_config)
    run_model(config)
    # get_eval()