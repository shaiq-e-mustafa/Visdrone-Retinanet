from torch.utils.data import DataLoader
from .dataset import DroneDetectionDataset
from sklearn.model_selection import train_test_split


def split_dataset(samples,
                  train_ratio=0.7,
                  val_ratio=0.2,
                  test_ratio=0.1,
                  seed=42):

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=seed
    )

    return train_samples, val_samples, test_samples


def collate_fn(batch):
    images, targets = zip(*batch)

    return list(images), list(targets)




def get_dataloaders(dataset_dir, batch_size=8, transforms=None):

    # Create full dataset
    full_dataset = DroneDetectionDataset(
        dataset_dir=dataset_dir,
        transforms=transforms
    )

    # Split samples
    train_samples, val_samples, test_samples = split_dataset(
        full_dataset.samples
    )

    train = DroneDetectionDataset(
        samples=train_samples,
        transforms=transforms
    )
    
    val = DroneDetectionDataset(
        samples=val_samples,
        transforms=transforms
    )
    test = DroneDetectionDataset(
        samples=test_samples,
        transforms=transforms
    )


    # Create loaders
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )


    return train_loader, val_loader, test_loader
