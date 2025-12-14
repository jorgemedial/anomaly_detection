import os
from pathlib import Path

import logging
import sys

import torch
from torch.utils.data import DataLoader, Dataset
import PIL
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

from torch.export import export

import numpy as np

from simplenet import Simplenet
from MVTecAD import MVTECTrainset

import json

with open("training_config.json", "r") as f:
    training_config = json.load(f)

root_logger = logging.getLogger("simplenet_training")
root_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

train_dataset = MVTECTrainset(category=training_config["category"])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    Simplenet = Simplenet().to(device)
    optimizer = torch.optim.Adam(Simplenet.parameters())

    best_loss = torch.tensor(np.inf)
    for i in range(training_config["total_epochs"]):
        for batch_idx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()

            data = data.to(device)
            z_score_correct, z_score_altered = Simplenet.forward(data)
            loss = Simplenet.loss(z_score_correct, z_score_altered)
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_loss = loss
                torch.save(Simplenet.state_dict(), "model.pth")
            print(f"Epoch: {i}, batch: {batch_idx}, batch_size: {batch_size}. \nLoss {loss} \n")
        print(f"Epoch: {i}. Best lost so far: {best_loss}")
    
   
