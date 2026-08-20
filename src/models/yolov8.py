import os
from src.utils.get_img_ann import list_stems
from src.utils.split_pairs import split_pairs
from src.utils.yolo_helper import process_split
from ultralytics import YOLO


def get_dataset_yolov8(root_dir, img_dir, ann_dir, train_split, test_split, val_split, dest_dir, mode, workers, seed, chunk_size, class_map):
   images_dir = os.path.join(root_dir, img_dir)
   annotations_dir = os.path.join(root_dir, ann_dir)

   pairs = list_stems(images_dir, annotations_dir)
   splits = split_pairs(pairs, train_split, val_split, test_split, seed)
 

   for split_name, split_pairs_list in splits.items():
        if len(split_pairs_list) == 0:
            continue
        process_split(split_name, split_pairs_list, dest_dir, mode, workers, chunk_size=chunk_size, class_map=class_map)

def train_yolo(yolo_model, yolo_config, epochs, image_size, save_dir, run_name, batch_size):
    model = YOLO(yolo_model)
    results = model.train(
        data=yolo_config,
        epochs=epochs,
        imgsz=image_size,
        project=save_dir,   # top-level folder
        name=run_name,             # subfolder for this run
        exist_ok=True,                   # overwrite instead of auto-incrementing if rerun
        batch=batch_size
    )