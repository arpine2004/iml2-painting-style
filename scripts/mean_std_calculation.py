import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
train_dir = PROJECT_ROOT / "data" / "train"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from iml2_painting_style.dataset import PaintingStyleDataset


def calculate_mean_std(dataset, batch_size=32, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += batch_pixels

    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_squared_sum / total_pixels) - (mean ** 2))

    return mean, std


temporary_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


train_dataset = PaintingStyleDataset(
    root_dir=train_dir,
    transform=temporary_transform
)

mean, std = calculate_mean_std(train_dataset)

print("Mean:", mean)
print("Std:", std)