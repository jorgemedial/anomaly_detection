from pathlib import Path
import torchvision
from torch.utils.data import Dataset
import PIL
import os

class MVTECTestset(Dataset):
    def __init__(self,  category: str, anomaly_name: str, root = "./datasets/MVTecAD"):
        self.input_path = Path(root) / Path(category) / Path("test") / Path(anomaly_name)
        self.ground_truth_path = Path(root) / Path(category) / Path("ground_truth") / Path(anomaly_name)

        self.output_path = Path("./outputs") / Path(category) / Path(anomaly_name)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(248),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, index):
        img = PIL.Image.open(self.input_path / Path(f"{index:>03}.png")).convert("RGB")
        img = self.transformations(img)
        return img
    

    def filename_from_index(self, index: int):
        return self.output_path / Path(f"{index:>03}.png")
    
    
    
        
