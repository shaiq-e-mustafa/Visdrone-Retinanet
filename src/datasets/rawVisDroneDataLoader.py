"""
Builds train/val/test DataLoaders directly from raw VisDrone annotations
(dataset_dir/images + dataset_dir/annotations), reusing your existing
get_img_ann.list_stems() and split_pairs.split_pairs() utilities.

This is the dataloader for the RetinaNet/FCOS torchvision benchmark stack -
separate from your YOLO dataloader.py, since the target format and source
data (raw annotations vs. converted YOLO labels) are different.
"""

from torch.utils.data import DataLoader

from src.utils.get_img_ann import list_stems
from src.utils.split_pairs import split_pairs 
from src.datasets.rawVisDroneDataset import RawVisDroneDataset, collate_fn


def get_dataloaders(
    images_dir,
    annotations_dir,
    train_transform,
    val_transform,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    batch_size=4,
    workers=4,
):
    pairs = list_stems(images_dir, annotations_dir)
    splits = split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed)

    train_ds = RawVisDroneDataset(splits["train"], transforms=train_transform)
    val_ds = RawVisDroneDataset(splits["val"], transforms=val_transform)
    test_ds = RawVisDroneDataset(splits["test"], transforms=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader