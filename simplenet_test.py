import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from simplenet import Simplenet
from MVTecAD import MVTECTestset
import torch.nn.functional as F
import json

with open("training_config.json", "r") as f:
    config = json.load(f)



def process_output(output):
    anomaly_map = output.reshape((31, 31, 1))
    anomaly_map = anomaly_map.permute(2, 0, 1).unsqueeze(0)
    anomaly_map_upscaled = F.interpolate(anomaly_map, scale_factor=8, mode="nearest")
    anomaly_map_final = F.pad(anomaly_map_upscaled, (4, 4, 4, 4))
    return anomaly_map_final.squeeze(0).permute(1, 2, 0)
                 
def export_as_image(anomaly_map: torch.Tensor):
    # x_padded: (H, W, 1)
    arr = anomaly_map.squeeze(-1).cpu().numpy()     # → (H, W)

    # Normalize to 0–255 if values are not already in that range
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")         # "L" = 8-bit grayscale



if __name__ == "__main__":

    model_filepath = "model.pth"

    simplenet = Simplenet()
    simplenet.load_state_dict(torch.load(model_filepath))

    device = torch.device('cpu')
    simplenet.eval()
 
    cracked_dataset = MVTECTestset(category=config["category"], anomaly_name="crack")

    
    with torch.no_grad():
        for i in range(cracked_dataset.__len__()):
            data = cracked_dataset.__getitem__(i).unsqueeze(0)
            data = data.to(device)             
            outputs = simplenet(data)
            for output in outputs:
                anomaly_map = process_output(output)
                image = export_as_image(anomaly_map)
                filename = cracked_dataset.filename_from_index(i)
                image.save(filename)

    


        