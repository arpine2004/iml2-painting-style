from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch

img_mean = torch.tensor([0.5224, 0.4805, 0.4183])
img_std = torch.tensor([0.2556, 0.2468, 0.2495])

def get_simclr_transforms(image_size=64):
    """
    Strong augmentation pipeline for SimCLR.
    Both views are produced by applying this transform independently to the same image.
    """
    kernel_size = max(3, int(0.1 * image_size) | 1) 
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0),
                                     interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])