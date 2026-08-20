import os
from glob import glob
import cv2
from torch.utils.data import Dataset
import torch

class DroneDetectionDataset(Dataset):
    def __init__(self, dataset_dir=None, samples=None, transforms=None):

        self.dataset_dir = dataset_dir
        self.transforms = transforms

        if samples is not None:
            self.samples = samples

        else:
            self.dataset_dir = dataset_dir
            self.samples = self._load_dataset()
            
    
    def _load_dataset(self):
        images = glob(os.path.join(self.dataset_dir, 'images', "*"))

        img_ann_arr = []

        for file in images:
            f = file.split('\\')[-1]
            annotation_file = f.split('.')[0]
            annotation_file_name = annotation_file + '.txt'
            ann = os.path.join(self.dataset_dir, 'annotations', annotation_file_name)
            img_ann_arr.append((file, ann))
        return img_ann_arr
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        img_path, ann_path = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # Load annotations
        boxes = []
        labels = []

        with open(ann_path, "r") as f:

            for line in f:

                values = line.strip().split(",")

                x = int(values[0])
                y = int(values[1])
                w = int(values[2])
                h = int(values[3])

                category = int(values[5])


                # Ignore VisDrone background
                if category == 0:
                    continue


                boxes.append(
                    [
                        x,
                        y,
                        x+w,
                        y+h
                    ]
                )

                labels.append(category)


        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )


        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }


        if self.transforms:
            image, target = self.transforms(
                image,
                target
            )


        return image, target
