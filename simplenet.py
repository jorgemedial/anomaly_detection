import os
from pathlib import Path
import cv2

import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

from anomalib.data import MVTecAD

class MVTECTrainset(Dataset):
    def __init__(self,  category: str, root = "./datasets/MVTecAD"):
        self.path = Path(root) / Path(category) / Path("train") / Path("good")
        
    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, index):
        img = cv2.imread(self.path / Path(f"{index:>03}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()/255
        img = img.permute(2, 0, 1)
        return img

train_dataset = MVTECTrainset(category="hazelnut")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

class Simplenet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Extract layers 2 and 3 of 
        return_nodes = {
            "layer2": "feat2",
            "layer3": "feat3",
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Sabemos que layer3 de ResNet50 tiene 1024 canales
        feature_channel_size = 512

        self.sigma_noise = 0.1
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=4, stride=1, padding=2)
        self.conv1x1 = torch.nn.Conv2d(feature_channel_size, feature_channel_size, kernel_size=1)

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(in_features=feature_channel_size, out_features=feature_channel_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=feature_channel_size, out_features=feature_channel_size),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        batch_size, _, h, w = x.shape
        out = self.backbone(x)   # dict con las keys que hemos definido
        feat2 = out["feat2"]

        pooled_features = self.avg_pool(feat2)
        adapted_features = self.conv1x1(pooled_features).permute(0, 2, 3, 1).reshape(-1, 512) 
        altered_features = adapted_features.detach().clone()             

        z_scores_correct = self.discriminator(adapted_features)

        if self.training:
            altered_features =  torch.randn_like(altered_features) * self.sigma_noise + adapted_features
        z_scores_altered = self.discriminator(altered_features)

        return z_scores_correct, z_scores_altered
    
    @staticmethod
    def loss(z_scores_correct, z_scores_altered):
        return (F.relu(0.5 - z_scores_correct) + F.relu(-0.5 + z_scores_altered)).sum()
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    simplenet = Simplenet().to(device)
    optimizer = torch.optim.Adam(simplenet.parameters())

    for i in range(1000):
        for batch_idx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()

            data = data.to(device)
            z_score_correct, z_score_altered = simplenet.forward(data)
            loss = simplenet.loss(z_score_correct, z_score_altered)

            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print(loss)
   
