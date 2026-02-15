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

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

file_handler = logging.FileHandler("./log.txt", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

root_logger.addHandler(handler)
root_logger.addHandler(file_handler)

train_dataset = MVTECTrainset(category=training_config["category"])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    Simplenet = Simplenet().to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": Simplenet.conv1x1.parameters(), "lr": 0.0001, "weight_decay":0.00001},
            {"params": Simplenet.discriminator.parameters(), "lr": 0.0002, "weight_decay":0.00001},
        ]
    )

    best_epoch_loss = torch.tensor(np.inf)
    epoch_loss = 0
    no_of_batches = 0
    for i in range(training_config["total_epochs"]):
        for batch_idx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()

            data = data.to(device)
            z_score_correct, z_score_altered = Simplenet.forward(data)
            loss = Simplenet.loss(z_score_correct, z_score_altered)
            loss.backward()
            optimizer.step()

            epoch_loss += loss 
            no_of_batches += 1
            root_logger.info(f"Epoch: {i}, batch: {batch_idx}, batch_size: {batch_size}. \nLoss {loss} \n")

        if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                torch.save(Simplenet.state_dict(), "model.pth")
        root_logger.info(f"Epoch: {i}. Best lost so far: {best_epoch_loss/no_of_batches}")
    
   
