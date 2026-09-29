"""serving.py - loads the deployment bundle (5-fold ensemble + U-Net) and renders explanation overlays."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / 'models' / 'deploy'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'classification'))
from backbones import EnsembleLogProb, load_member
from decision_support import DecisionSupport
from explain_wrappers import make_mask_fn, make_cam_fn


def _clf_transform(size):
    spec = importlib.util.spec_from_file_location('clf_transforms', ROOT / 'src' / 'classification' / 'transforms.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_val_transform(size)


class Service:
    def __init__(self, bundle_dir=DEFAULT_BUNDLE, device='cpu'):
        bd = Path(bundle_dir)
        b = json.loads((bd / 'bundle.json').read_text())
        dev = torch.device(device)
        members = [load_member(str(bd / f), dev) for f in b['members']]
        model = EnsembleLogProb(members).eval()
        tf = _clf_transform(b['image_size'])
        mask_fn = None
        if b.get('unet') and (bd / b['unet']).exists():
            mask_fn = make_mask_fn(str(bd / b['unet']), dev, src_dir=str(ROOT / 'src' / 'segmentation'))
        cam_fn = make_cam_fn(members[0], tf, dev)          # Grad-CAM from the first ensemble member
        args = (model, b['class_names'], tf, dev, b['temperature'], b['calibrated_conf_threshold'])
        self.full = DecisionSupport(*args, mask_fn=mask_fn, cam_fn=cam_fn)
        self.fast = DecisionSupport(*args)
        self.info = {k: b[k] for k in ('tag', 'model_name', 'image_size', 'temperature', 'calibrated_conf_threshold')}

    def predict(self, pil, explain=True):
        return (self.full if explain else self.fast).predict(pil)


def render(pil, result, size=224):
    base = np.array(pil.resize((size, size)))
    out = {}
    if result.get('mask') is not None:
        m = np.array(Image.fromarray((result['mask'] * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)) > 127
        edge = (m != np.roll(m, 1, 0)) | (m != np.roll(m, 1, 1))
        edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        o = base.copy()
        o[edge] = (0, 255, 0)
        out['mask'] = Image.fromarray(o)
    if result.get('heatmap') is not None:
        h = np.array(Image.fromarray((result['heatmap'] * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
        rgb = (colormaps['jet'](h)[..., :3] * 255).astype(np.uint8)
        out['heatmap'] = Image.fromarray((0.6 * base + 0.4 * rgb).astype(np.uint8))
    return out
