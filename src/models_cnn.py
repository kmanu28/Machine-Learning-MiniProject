"""
Convolutional Neural Network Architectures for Airbus Ship Detection:
1. Simple CNN (SIM-CNN): Custom 4-layer CNN trained from scratch (CS229 Fig 3)
   - Input: (3, 256, 256)
   - Conv2d(3->16) + MaxPool -> (16, 128, 128)
   - Conv2d(16->32) + MaxPool -> (32, 64, 64)
   - Conv2d(32->64) + MaxPool -> (64, 32, 32)
   - Conv2d(64->128) + MaxPool -> (128, 16, 16)
   - FC(32768 -> 128) -> Dropout(0.5) -> FC(128 -> 1)
2. Transfer Learning CNN (TL-CNN): Pretrained backbone with fine-tuning head (CS229 Fig 2)
   - Pretrained ImageNet DenseNet/MobileNet backbone (frozen feature extractor)
   - BatchNorm -> Dropout -> FC(128) -> FC(1)
"""

import os
import sys
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
except ImportError:
    torch = None
    nn = None
    models = None


if torch is not None:

    class SimpleCNN(nn.Module):
        """
        SIM-CNN: 4-layer Convolutional Neural Network trained from scratch.
        Mirrors exact architecture from Stanford CS229 paper.
        """
        def __init__(self, in_channels: int = 3, num_classes: int = 1):
            super(SimpleCNN, self).__init__()
            
            # Convolutional Block 1
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128
            )
            
            # Convolutional Block 2
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
            )
            
            # Convolutional Block 3
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
            )
            
            # Convolutional Block 4
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16
            )
            
            # Dense Classifier Block (128 channels * 16 * 16 = 32768)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            logits = self.classifier(x)
            return logits


    class TransferLearningCNN(nn.Module):
        """
        TL-CNN: Transfer Learning with Pretrained Backbone (DenseNet / MobileNet / ResNet).
        Freezes feature extractor layers and trains dense classification head.
        """
        def __init__(self, backbone: str = "densenet169", pretrained: bool = True, num_classes: int = 1):
            super(TransferLearningCNN, self).__init__()
            self.backbone_name = backbone

            if backbone == "densenet169":
                try:
                    base = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None)
                except Exception as exc:
                    if not pretrained:
                        raise
                    print(f"Warning: DenseNet-169 weights unavailable ({exc}); using random initialization.")
                    base = models.densenet169(weights=None)
                in_features = base.classifier.in_features
                self.features = base.features
            elif backbone == "densenet121":
                base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
                in_features = base.classifier.in_features
                self.features = base.features
            elif backbone == "mobilenet_v2":
                base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
                in_features = base.last_channel
                self.features = base.features
            elif backbone == "resnet18":
                base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
                in_features = base.fc.in_features
                self.features = nn.Sequential(*list(base.children())[:-1])
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            # Freeze backbone weights for transfer learning
            for param in self.features.parameters():
                param.requires_grad = False

            # Custom classification head
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=0.4),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.features(x)
            logits = self.head(feat)
            return logits

else:
    class SimpleCNN:
        pass

    class TransferLearningCNN:
        pass


def count_trainable_parameters(model) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    if torch is None:
        return 0, 0
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
