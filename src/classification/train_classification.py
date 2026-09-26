"""
train_classification.py - DermaAI Classification Training Script

Generic training loop that works with any model from models.py
(baseline_cnn, resnet50, efficientnet_b0). Tracks accuracy, macro-F1,
per-class precision/recall/F1, and macro ROC-AUC every epoch.
Saves the best checkpoint by validation macro-F1 (not accuracy, since
accuracy is misleading under the 58x class imbalance).

Supports configurable loss_type: 'ce' (unweighted baseline), 'weighted_ce'
(class-weighted CrossEntropyLoss), 'focal' (Focal Loss) for Part 5's
class imbalance handling experiments.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, classification_report
)

sys.path.insert(0, '/kaggle/working/repo/src/classification')

from dataset import ClassificationDataset, CLASS_NAMES
from transforms import get_train_transform, get_val_transform
from models import get_model
from losses import compute_class_weights, FocalLoss


def build_criterion(loss_type, train_csv, class_names, device, focal_gamma=2.0):
    """
    loss_type: 'ce' (unweighted CrossEntropyLoss, the naive baseline),
               'weighted_ce' (class-weighted CrossEntropyLoss),
               'focal' (Focal Loss, optionally with class weights as alpha)
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        weights = compute_class_weights(train_csv, class_names).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_type == 'focal':
        weights = compute_class_weights(train_csv, class_names).to(device)
        return FocalLoss(alpha=weights, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from 'ce', 'weighted_ce', 'focal'.")


def compute_metrics(y_true, y_pred, y_probs, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    try:
        macro_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro', labels=range(len(class_names)))
    except ValueError:
        macro_auc = float('nan')

    per_class = {
        class_names[i]: {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i]),
        } for i in range(len(class_names))
    }

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_auc': macro_auc,
        'per_class': per_class,
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs), CLASS_NAMES)
    return avg_loss, metrics


def run_training(
    model_name,
    train_csv, val_csv,
    output_dir,
    image_size=224,
    batch_size=32,
    num_epochs=20,
    lr=1e-4,
    num_workers=2,
    loss_type='ce',
    focal_gamma=2.0,
    experiment_name=None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = experiment_name or f"{model_name}_{loss_type}"
    print(f"Training '{exp_name}' (model={model_name}, loss={loss_type}) on device: {device}")

    train_ds = ClassificationDataset(train_csv, image_size=image_size, transform=get_train_transform(image_size))
    val_ds = ClassificationDataset(val_csv, image_size=image_size, transform=get_val_transform(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = get_model(model_name, num_classes=len(CLASS_NAMES)).to(device)
    criterion = build_criterion(loss_type, train_csv, CLASS_NAMES, device, focal_gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        scheduler.step(val_metrics['macro_f1'])
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f} val_auc={val_metrics['macro_auc']:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_macro_f1': train_metrics['macro_f1'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_macro_auc': val_metrics['macro_auc'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': elapsed,
        })

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            checkpoint_path = os.path.join(output_dir, f'{exp_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'loss_type': loss_type,
                'model_state_dict': model.state_dict(),
                'val_macro_f1': best_val_f1,
                'val_accuracy': val_metrics['accuracy'],
                'per_class_metrics': val_metrics['per_class'],
            }, checkpoint_path)
            print(f"  -> New best model saved (val_macro_f1={best_val_f1:.4f})")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, f'{exp_name}_training_history.csv'), index=False)
    print(f"\nTraining complete for {exp_name}. Best val_macro_f1={best_val_f1:.4f}")

    return history_df, best_val_f1
