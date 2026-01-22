import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=16, num_workers=4):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    torch.backends.cudnn.benchmark = True

    BASE_DIR = r"D:\DL Project\lung_cancer_project\dataset_split"

    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR   = os.path.join(BASE_DIR, "val")
    TEST_DIR  = os.path.join(BASE_DIR, "test")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(TEST_DIR, transform=val_test_transforms)

    print("Classes:", train_dataset.classes)
    print("Number of classes:", len(train_dataset.classes))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# OPTIONAL: standalone sanity test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("✅ DataLoader working correctly")
