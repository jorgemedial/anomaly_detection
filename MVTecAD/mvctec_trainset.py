from pathlib import Path
import torchvision
from torch.utils.data import Dataset
import PIL
import os

class MVTECTrainset(Dataset):
    def __init__(self,  category: str, root = "./datasets/MVTecAD"):
        self.path = Path(root) / Path(category) / Path("train") / Path("good")
        self.transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(248),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, index):
        img = PIL.Image.open(self.path / Path(f"{index:>03}.png")).convert("RGB")
        img = self.transformations(img)
        return img
