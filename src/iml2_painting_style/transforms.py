from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch

img_mean = torch.tensor([0.5224, 0.4805, 0.4183])
img_std = torch.tensor([0.2556, 0.2468, 0.2495])

def get_transforms(split, image_size=64):
    split = split.lower()

    base_transforms = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])

    elif split in {"val", "valid", "validation", "test"}:
        return transforms.Compose(base_transforms)

    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")