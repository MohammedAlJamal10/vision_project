import torch.nn as nn
from torchvision import models


class MobileNetV2Lite(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.backbone = models.mobilenet_v2(weights="DEFAULT")

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.freeze_backbone()

    def freeze_backbone(self):

        for param in self.backbone.features.parameters():
            param.requires_grad = False

        self.freeze_batchnorm()

    def unfreeze_last_layers(self, ratio=0.25):

        feature_blocks = list(self.backbone.features)
        start_index = int(len(feature_blocks) * (1 - ratio))

        for block in feature_blocks[start_index:]:

            for module in block.modules():

                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

                    for param in module.parameters():
                        param.requires_grad = False

                    continue

                for param in module.parameters(recurse=False):
                    param.requires_grad = True

        self.freeze_batchnorm()

    def freeze_batchnorm(self):

        for module in self.backbone.features.modules():

            if isinstance(module, nn.BatchNorm2d):
                module.eval()

                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):

        return self.backbone(x)
