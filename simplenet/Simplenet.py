import torch, torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

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
        feature_channel_size = 1536

        self.sigma_noise = 0.015
        self.conv1x1 = torch.nn.Conv2d(feature_channel_size, feature_channel_size, kernel_size=1)

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(in_features=feature_channel_size, out_features=feature_channel_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=feature_channel_size, out_features=1)
        )

    def forward(self, x):
        batch_size, _, h, w = x.shape
        out = self.backbone(x)   # dict con las keys que hemos definido
        feat2, feat3 = out["feat2"], out["feat3"]

                
        # Upsample feat3 to feat2 size
        feat3_up = F.interpolate(feat3, size=feat2.shape[2:], mode="bilinear", align_corners=False)

        merged_feats = torch.cat([feat2, feat3_up], dim=1)
        
        num_channels = merged_feats.shape[1]

        # pooled_features = self.avg_pool(feat2)
        adapted_features = self.conv1x1(merged_feats).permute(0, 2, 3, 1).reshape(-1, num_channels) 
        altered_features = adapted_features.detach().clone()             

        z_scores_correct = self.discriminator(adapted_features)

        if self.training:
            altered_features =  torch.randn_like(altered_features) * self.sigma_noise + adapted_features
        z_scores_altered = self.discriminator(altered_features)

        return z_scores_correct, z_scores_altered
    
    @staticmethod
    def loss(z_scores_correct, z_scores_altered):
        correct_score = F.relu(0.5 - z_scores_correct).sum()
        altered_score = F.relu(-0.5 + z_scores_altered).sum()
        print(f"Correct score: {correct_score}")
        print(f"Altered score: {altered_score}")
        return  correct_score + altered_score

