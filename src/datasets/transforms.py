import albumentations as A
from albumentations.pytorch import ToTensorV2

LONG_SIDE = 960  # tune to VRAM — try 960/1024/1280

train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=LONG_SIDE),
        A.PadIfNeeded(
            min_height=LONG_SIDE, min_width=LONG_SIDE,
            border_mode=0, value=0,
            position="top_left",  # keeps bbox math simple — no centering offset to track
        ),
        A.HorizontalFlip(p=0.5),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
)

val_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=LONG_SIDE),
        A.PadIfNeeded(min_height=LONG_SIDE, min_width=LONG_SIDE, border_mode=0, value=0, position="top_left"),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
)