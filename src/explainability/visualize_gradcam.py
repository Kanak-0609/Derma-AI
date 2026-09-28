"""
visualize_gradcam.py - DermaAI Grad-CAM figures

predict_all: batch predictions over a dataset (no gradients).
make_gradcam_figure: rows of [original | overlay for predicted class | overlay for true class].
"""

import os
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from gradcam import GradCAM, overlay_heatmap_on_image


def load_display_image(path, size=224):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size))


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    labels, preds, probs = [], [], []
    for images, y in loader:
        p = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
        probs.append(p)
        preds.extend(p.argmax(axis=1).tolist())
        labels.extend(y.tolist())
    return np.array(labels), np.array(preds), np.vstack(probs)


def make_gradcam_figure(model, dataset, indices, class_names, device, output_path, title):
    cam_extractor = GradCAM(model, model.features[-1])
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image, true_label = dataset[idx]
        x = image.unsqueeze(0).to(device)

        cam_pred, pred_class, pred_prob = cam_extractor.generate(x)  # explains predicted class
        cam_true, _, true_prob = cam_extractor.generate(x, target_class=true_label)  # explains true class

        raw = load_display_image(dataset.df.iloc[idx][dataset.path_col])

        axes[row, 0].imshow(raw)
        axes[row, 0].set_title(f"true: {class_names[true_label]}", fontsize=9)
        axes[row, 1].imshow(overlay_heatmap_on_image(raw, cam_pred))
        axes[row, 1].set_title(f"pred: {class_names[pred_class]} (p={pred_prob:.2f})", fontsize=9)
        axes[row, 2].imshow(overlay_heatmap_on_image(raw, cam_true))
        axes[row, 2].set_title(f"evidence for {class_names[true_label]} (p={true_prob:.2f})", fontsize=9)
        for a in axes[row]:
            a.axis('off')

    cam_extractor.remove_hooks()
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130)
    plt.close(fig)
    print("Saved:", output_path)
