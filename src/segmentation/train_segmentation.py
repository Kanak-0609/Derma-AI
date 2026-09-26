"""
train_segmentation.py - DermaAI U-Net Training Script

Trains the U-Net on the ISIC2018 Task 1 segmentation data, tracks
Dice/IoU/Precision/Recall each epoch, saves the best checkpoint by
validation Dice, and logs history to CSV.
"""

import os
import sys
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append('/kaggle/working/repo/src/segmentation')
sys.path.append('/kaggle/working/repo/src/evaluation')

from dataset import SegmentationDataset
from transforms import get_train_transform, get_val_transform
from unet import UNet
from losses_metrics import BCEDiceLoss, compute_segmentation_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        batch_metrics = compute_segmentation_metrics(logits, masks)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k] * images.size(0)

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        batch_metrics = compute_segmentation_metrics(logits, masks)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k] * images.size(0)

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return avg_loss, avg_metrics


def run_training(
    train_csv, val_csv,
    output_dir,
    image_size=256,
    batch_size=16,
    num_epochs=30,
    lr=1e-4,
    num_workers=2,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_ds = SegmentationDataset(train_csv, image_size=image_size, transform=get_train_transform(image_size))
    val_ds = SegmentationDataset(val_csv, image_size=image_size, transform=get_val_transform(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_dice = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['dice'])
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_loss': val_loss,
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': elapsed,
        })

        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            checkpoint_path = os.path.join(output_dir, 'unet_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': best_val_dice,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_dice={best_val_dice:.4f})")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    print(f"\nTraining complete. Best val_dice={best_val_dice:.4f}")
    print(f"Checkpoint + history saved to: {output_dir}")

    return history_df, best_val_dice
