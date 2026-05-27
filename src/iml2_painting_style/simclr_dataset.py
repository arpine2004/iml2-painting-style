from PIL import Image
from torch.utils.data import Dataset
from .dataset import PaintingStyleDataset


class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.base = PaintingStyleDataset(root_dir, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_path, label = self.base.samples[idx]
        image = Image.open(img_path).convert('RGB')
        view1 = self.transform(image)
        view2 = self.transform(image)     
        return view1, view2, label