"""
decision_support.py - DermaAI decision-support layer

predict(image) -> dict with predicted class, calibrated probabilities,
referral flag, plain-language message, and optional mask / heatmap.
Research decision-support only. NOT a diagnostic device.
"""
import json
import numpy as np
import torch
from PIL import Image

DISCLAIMER = ("This is a research tool, not a medical diagnosis. "
              "Only a qualified clinician can diagnose a skin lesion.")

FULL_NAMES = {'akiec': 'actinic keratosis / intraepithelial carcinoma', 'bcc': 'basal cell carcinoma',
              'bkl': 'benign keratosis-like lesion', 'df': 'dermatofibroma',
              'mel': 'melanoma', 'nv': 'melanocytic nevus (mole)', 'vasc': 'vascular lesion'}


class DecisionSupport:
    def __init__(self, model, class_names, transform, device,
                 temperature, conf_threshold, mask_fn=None, cam_fn=None):
        self.model = model.eval()
        self.class_names = list(class_names)
        self.transform = transform
        self.device = device
        self.T = float(temperature)
        self.conf_threshold = float(conf_threshold)
        self.mask_fn = mask_fn   # callable(PIL image) -> 2D numpy mask, optional
        self.cam_fn = cam_fn     # callable(PIL image, class_idx) -> 2D numpy heatmap, optional

    @classmethod
    def from_files(cls, model, class_names, transform, device,
                   temperature_json, mask_fn=None, cam_fn=None):
        with open(temperature_json) as f:
            cfg = json.load(f)
        return cls(model, class_names, transform, device,
                   cfg['temperature'], cfg['calibrated_conf_threshold'], mask_fn, cam_fn)

    def _to_tensor(self, pil):
        arr = np.array(pil)
        try:                                   # albumentations style
            out = self.transform(image=arr)['image']
        except TypeError:                      # torchvision style
            out = self.transform(pil)
        return out.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image):
        pil = Image.open(image).convert('RGB') if isinstance(image, str) else image.convert('RGB')
        logits = self.model(self._to_tensor(pil))
        probs = torch.softmax(logits / self.T, dim=1)[0].cpu().numpy()  # temperature on logits
        idx = int(probs.argmax())
        conf = float(probs[idx])
        referred = conf < self.conf_threshold
        name = self.class_names[idx]

        if referred:
            msg = (f"Low confidence ({conf:.0%}). Most likely {FULL_NAMES.get(name, name)}, "
                   f"but the system is not sure. Recommend review by a dermatologist.")
        else:
            msg = (f"Most likely {FULL_NAMES.get(name, name)} (confidence {conf:.0%}). "
                   f"Have any changing or worrying lesion checked by a clinician.")
        if name == 'mel':
            msg += " Melanoma is the top prediction, so a clinical check is advised."

        result = {'predicted_class': name, 'confidence': conf,
                  'probabilities': {c: float(p) for c, p in zip(self.class_names, probs)},
                  'referred_for_review': bool(referred), 'message': msg + " " + DISCLAIMER,
                  'mask': None, 'heatmap': None}
        if self.mask_fn is not None:
            result['mask'] = self.mask_fn(pil)
        if self.cam_fn is not None:
            result['heatmap'] = self.cam_fn(pil, idx)
        return result
