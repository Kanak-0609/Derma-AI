"""
evaluate_segmentation.py - DermaAI Segmentation Evaluation

Loads the best U-Net checkpoint, evaluates it on the held-out test set,
and generates side-by-side visualizations (image / ground truth / prediction).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append('/kaggle/working/repo/src/segmentation')
sys.path.append('/kaggle/working/repo/src/evaluation')

from dataset import SegmentationDataset
from transforms import get_val_transform
from unet import UNet
from losses_metrics import BCEDiceLoss, compute_segmentation_metrics


def load_model(checkpoint_path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_dice={checkpoint['val_dice']:.4f})")
    return model


@torch.no_grad()
def evaluate_test_set(model, test_csv, device, image_size=256, batch_size=16):
    test_ds = SegmentationDataset(test_csv, image_size=image_size, transform=get_val_transform(image_size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = BCEDiceLoss(bce_weight=0.5)
    running_loss = 0.0
    running_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        batch_metrics = compute_segmentation_metrics(logits, masks)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k] * images.size(0)

    n = len(test_loader.dataset)
    avg_loss = running_loss / n
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return avg_loss, avg_metrics


def denormalize_image(tensor_image):
    """Reverses ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor_image.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


@torch.no_grad()
def visualize_predictions(model, test_csv, device, output_path, image_size=256, num_samples=6, threshold=0.5):
    test_ds = SegmentationDataset(test_csv, image_size=image_size, transform=get_val_transform(image_size))

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    indices = np.random.choice(len(test_ds), size=num_samples, replace=False)

    for row_idx, idx in enumerate(indices):
        image, mask = test_ds[idx]
        image_batch = image.unsqueeze(0).to(device)

        logits = model(image_batch)
        pred = (torch.sigmoid(logits) > threshold).float().cpu().squeeze().numpy()

        img_display = denormalize_image(image)
        mask_display = mask.numpy()

        axes[row_idx, 0].imshow(img_display)
        axes[row_idx, 0].set_title('Original Image')
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(mask_display, cmap='gray')
        axes[row_idx, 1].set_title('Ground Truth Mask')
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(pred, cmap='gray')
        axes[row_idx, 2].set_title('Predicted Mask')
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved prediction visualization to: {output_path}")
