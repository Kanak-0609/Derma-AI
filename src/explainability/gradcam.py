"""
gradcam.py - DermaAI Grad-CAM Explainability

Grad-CAM (Selvaraju et al., 2017) implemented from scratch. Captures the target
layer's activations and the gradient of the class score with respect to them,
then builds a class-discriminative heatmap.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        # Tensor-level hook: avoids in-place/view problems of module backward hooks
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: preprocessed image, shape (1, C, H, W)
        target_class: class index to explain; None = model's top prediction
        Returns: (cam HxW in [0,1], target_class, probability of target_class)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        prob = probs[0, target_class].item()

        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class, prob

    def remove_hooks(self):
        self.forward_handle.remove()


def overlay_heatmap_on_image(image_rgb_uint8, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    h, w = image_rgb_uint8.shape[:2]
    cam_uint8 = np.uint8(255 * cv2.resize(cam, (w, h)))
    heatmap = cv2.cvtColor(cv2.applyColorMap(cam_uint8, colormap), cv2.COLOR_BGR2RGB)
    overlay = image_rgb_uint8.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
