import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from simplenet import Simplenet
from MVTecAD import MVTECTestset
import torch.nn.functional as F
import json
import cv2

BLEND_RATIO = 0.3

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
    arr = np.maximum(0, arr - 0.5) + 1 # values below 0.5 are not considered anomalies due to the loss function definition.
    arr = np.minimum(255, np.log(arr) * 1000).astype(np.uint8) # The arr values are scaled up for visualization. Saturates at arr = 0.79
    
    return cv2.applyColorMap(arr, cv2.COLORMAP_JET)      

if __name__ == "__main__":

    model_filepath = "model.pth"

    simplenet = Simplenet()
    simplenet.load_state_dict(torch.load(model_filepath))

    device = torch.device('cpu')
    simplenet.eval()
 
    anomalous_dataset = MVTECTestset(category=config["category"], anomaly_name=config["anomaly_type"])

    
    with torch.no_grad():
        for i in range(anomalous_dataset.__len__()):
            frame = anomalous_dataset.__getitem__(i, transformed=False)
            data = anomalous_dataset.__getitem__(i).unsqueeze(0)
            data = data.to(device)             
            outputs = simplenet(data)
            for output in outputs:
                # Original image
                frame = anomalous_dataset.get_image_as_frame(i)
                shape = frame.shape[:2]

                # Heatmap
                anomaly_map = process_output(output)
                anomaly_heatmap = export_as_image(anomaly_map)
                anomaly_heatmap = cv2.resize(anomaly_heatmap, shape)

                # Fusion
                blend = BLEND_RATIO * anomaly_heatmap + (1 - BLEND_RATIO) * frame
                full_picture = np.concat([blend, anomaly_heatmap], axis=1)

                filename = anomalous_dataset.filename_from_index(i)
                cv2.imwrite(filename, full_picture)
    


        