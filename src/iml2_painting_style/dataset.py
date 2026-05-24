from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

class PaintingStyleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform 

        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {self.root_dir}")
        
        self.classes = sorted([folder.name for folder 
                               in self.root_dir.iterdir() if folder.is_dir()])
        
        if len(self.classes) == 0:
            raise ValueError(f"No class subdirectories found in: {self.root_dir}")
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name 
            for img_path in class_dir.rglob('*'):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No image files found in: {self.root_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

