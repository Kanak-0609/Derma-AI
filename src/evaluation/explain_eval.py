"""explain_eval.py - quantitative checks for Grad-CAM: localisation, deletion test, randomisation check."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from common import apply_tf


def _mask_at(mask_fn, pil, shape):
    m = mask_fn(pil)
    return np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST)) > 127


def evaluate_cam(model, paths, tf, mask_fn, make_cam_fn, device, n=150, seed=0):
    rng = np.random.default_rng(seed)
    paths = list(rng.choice(paths, size=min(n, len(paths)), replace=False))
    cam_fn = make_cam_fn(model, tf, device)
    fracs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    inside, area, hit, c_cam, c_rnd = [], [], [], [], []
    for p in paths:
        pil = Image.open(p).convert('RGB')
        x = apply_tf(tf, pil).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = int(model(x).argmax(1))
        cam = cam_fn(pil, idx)
        m = _mask_at(mask_fn, pil, cam.shape)
        inside.append(float((cam * m).sum() / (cam.sum() + 1e-8)))
        area.append(float(m.mean()))
        hit.append(bool(m.flat[int(cam.argmax())]))
        H, W_ = cam.shape
        variants = []
        for order in (np.argsort(-cam.ravel()), rng.permutation(cam.size)):
            for f in fracs:
                drop = np.zeros(cam.size, bool)
                drop[order[:int(f * cam.size)]] = True
                v = x.clone()
                v[:, :, torch.from_numpy(drop.reshape(H, W_)).to(device)] = 0.0
                variants.append(v)
        with torch.no_grad():
            pr = torch.softmax(model(torch.cat(variants)), 1)[:, idx].cpu().numpy()
        c_cam.append(pr[:len(fracs)])
        c_rnd.append(pr[len(fracs):])
    loc = pd.DataFrame({'metric': ['cam_energy_inside_mask', 'mask_area_fraction', 'ratio_inside_vs_area',
                                   'pointing_hit_rate'],
                        'value': [np.mean(inside), np.mean(area), np.mean(inside) / np.mean(area), np.mean(hit)]})
    dele = pd.DataFrame({'fraction_removed': fracs, 'prob_after_cam_deletion': np.mean(c_cam, 0),
                         'prob_after_random_deletion': np.mean(c_rnd, 0)})
    return loc, dele


def randomisation_check(model, model_rand, paths, tf, make_cam_fn, device, n=50, seed=0):
    """model_rand must be a separate copy of the model; its Linear layers are re-initialised here."""
    for mod in model_rand.modules():
        if isinstance(mod, nn.Linear):
            mod.reset_parameters()
    rng = np.random.default_rng(seed)
    paths = list(rng.choice(paths, size=min(n, len(paths)), replace=False))
    cam_a, cam_b = make_cam_fn(model, tf, device), make_cam_fn(model_rand, tf, device)
    rhos = []
    for p in paths:
        pil = Image.open(p).convert('RGB')
        x = apply_tf(tf, pil).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = int(model(x).argmax(1))
        a, b = cam_a(pil, idx), cam_b(pil, idx)
        if a.std() > 0 and b.std() > 0:
            rhos.append(spearmanr(a.ravel(), b.ravel()).correlation)
    return float(np.nanmean(rhos)), len(rhos)
