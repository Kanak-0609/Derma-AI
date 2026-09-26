"""
models.py - DermaAI Classification Models

Three architectures sharing a common interface (num_classes=7):
    1. BaselineCNN   - shallow CNN trained from scratch, no pretraining
    2. build_resnet50    - ImageNet-pretrained ResNet50, fine-tuned
    3. build_efficientnet_b0 - ImageNet-pretrained EfficientNet-B0, fine-tuned
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    """A simple CNN trained from scratch - no pretraining, no transfer learning.
    Serves as the 'naive baseline' data point in the model comparison table."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.AdaptiveAvgPool2d(1),  # -> (256, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet50(num_classes=7, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=7, pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY = {
    'baseline_cnn': lambda num_classes=7: BaselineCNN(num_classes=num_classes),
    'resnet50': lambda num_classes=7: build_resnet50(num_classes=num_classes, pretrained=True),
    'efficientnet_b0': lambda num_classes=7: build_efficientnet_b0(num_classes=num_classes, pretrained=True),
}


def get_model(name, num_classes=7):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        model = get_model(name)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output shape {out.shape}, params={n_params:,}")
