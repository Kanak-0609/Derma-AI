"""
evaluate_classification.py - DermaAI Classification Test Set Evaluation

Loads each trained model's best checkpoint, evaluates on the held-out
test set, and builds the final comparison table (Accuracy, macro-Precision,
macro-Recall, macro-F1, macro-ROC-AUC) plus a full per-class breakdown
and confusion matrix for each model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/kaggle/working/repo/src/classification')

from dataset import ClassificationDataset, CLASS_NAMES
from transforms import get_val_transform
from models import get_model


@torch.no_grad()
def evaluate_model_on_test(experiment_name, checkpoint_path, test_csv, device, image_size=224, batch_size=32):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Prefer the architecture name stored inside the checkpoint; fall back to the
    # experiment_name itself for older checkpoints that didn't store this field.
    model_name = checkpoint.get('model_name', experiment_name)

    model = get_model(model_name, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_ds = ClassificationDataset(test_csv, image_size=image_size, transform=get_val_transform(image_size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    try:
        macro_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro', labels=range(len(CLASS_NAMES)))
    except ValueError:
        macro_auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    per_class = {
        CLASS_NAMES[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        } for i in range(len(CLASS_NAMES))
    }

    summary = {
        'experiment': experiment_name,
        'architecture': model_name,
        'checkpoint_epoch': checkpoint['epoch'],
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'macro_auc': float(macro_auc),
    }

    return summary, per_class, cm


def plot_confusion_matrix(cm, class_names, experiment_name, output_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_comparison_table(experiments_and_checkpoints, test_csv, device, output_dir):
    """
    experiments_and_checkpoints: dict of {experiment_name: checkpoint_path}
    The actual model architecture is read from inside each checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []
    all_per_class = {}

    for experiment_name, checkpoint_path in experiments_and_checkpoints.items():
        print(f"Evaluating {experiment_name} on test set...")
        summary, per_class, cm = evaluate_model_on_test(experiment_name, checkpoint_path, test_csv, device)
        all_summaries.append(summary)
        all_per_class[experiment_name] = per_class

        cm_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrix.png')
        plot_confusion_matrix(cm, CLASS_NAMES, experiment_name, cm_path)
        print(f"  Test accuracy={summary['accuracy']:.4f}, macro_f1={summary['macro_f1']:.4f}, macro_auc={summary['macro_auc']:.4f}")

    comparison_df = pd.DataFrame(all_summaries)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison_table.csv'), index=False)

    with open(os.path.join(output_dir, 'per_class_metrics.json'), 'w') as f:
        json.dump(all_per_class, f, indent=2)

    print("\n=== FINAL COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))

    return comparison_df, all_per_class
