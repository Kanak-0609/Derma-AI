"""robustness.py - shortcut probe, corruption robustness, subgroup analysis."""
import io
import numpy as np, pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import f1_score
from common import CLASS_NAMES, MEL, predict_paths


def _summ(y, P):
    pred = P.argmax(1)
    rec = [(pred[y == c] == c).mean() for c in np.unique(y)]
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(f1_score(y, pred, average='macro')),
            'balanced_acc': float(np.mean(rec)),
            'recall_mel': float((pred[y == MEL] == MEL).mean()) if (y == MEL).any() else float('nan')}


def _sample(csv, n, seed):
    df = pd.read_csv(csv)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df, df.diagnosis.map({c: i for i, c in enumerate(CLASS_NAMES)}).values


def _mask_fill(pil, m, keep):
    a = np.array(pil).astype(np.float32)
    mf = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(pil.size, Image.NEAREST)) > 127
    fill = a.mean((0, 1))
    sel = mf if keep == 'lesion' else ~mf
    return Image.fromarray(np.where(sel[..., None], a, fill[None, None, :]).astype(np.uint8))


def shortcut_probe(model, val_csv, tf, mask_fn, device, n=600, seed=0):
    df, y = _sample(val_csv, n, seed)
    paths = df.image_path.tolist()
    conds = {'original': None,
             'lesion_only (background removed)': lambda p: _mask_fill(p, mask_fn(p), 'lesion'),
             'background_only (lesion removed)': lambda p: _mask_fill(p, mask_fn(p), 'background')}
    return pd.DataFrame([{'condition': k, **_summ(y, predict_paths(model, paths, tf, device, pil_fn=f))}
                         for k, f in conds.items()])


def _jpeg(q):
    def f(p):
        b = io.BytesIO()
        p.save(b, 'JPEG', quality=q)
        b.seek(0)
        return Image.open(b).convert('RGB')
    return f


def _noise(s):
    def f(p):
        a = np.array(p).astype(np.float32) + np.random.default_rng(0).normal(0, s, np.array(p).shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return f


OPS = {'original': None,
       'blur_radius2': lambda p: p.filter(ImageFilter.GaussianBlur(2)),
       'darker_0.6': lambda p: ImageEnhance.Brightness(p).enhance(0.6),
       'brighter_1.5': lambda p: ImageEnhance.Brightness(p).enhance(1.5),
       'jpeg_quality20': _jpeg(20),
       'gaussian_noise_15': _noise(15)}


def robustness_probe(model, val_csv, tf, device, n=600, seed=1):
    df, y = _sample(val_csv, n, seed)
    paths = df.image_path.tolist()
    return pd.DataFrame([{'condition': k, **_summ(y, predict_paths(model, paths, tf, device, pil_fn=f))}
                         for k, f in OPS.items()])


def subgroup_table(oof, y, P, meta):
    d = oof[['image_id']].merge(meta[['image_id', 'age', 'sex', 'localization']], on='image_id', how='left')
    d['age_group'] = pd.cut(d.age, [0, 30, 50, 70, 120], labels=['<30', '30-49', '50-69', '70+'],
                            right=False).astype(object).fillna('unknown')
    d['sex'] = d.sex.fillna('unknown')
    d['localization'] = d.localization.fillna('unknown')
    pred = P.argmax(1)
    rows = []
    for col in ['sex', 'age_group', 'localization']:
        for val, idx in d.groupby(col).indices.items():
            yy, pp = y[idx], pred[idx]
            rec = [(pp[yy == c] == c).mean() for c in np.unique(yy)]
            nm = int((yy == MEL).sum())
            rows.append({'group': col, 'value': val, 'n': len(idx), 'accuracy': (pp == yy).mean(),
                         'balanced_acc': np.mean(rec), 'n_mel': nm,
                         'recall_mel': (pp[yy == MEL] == MEL).mean() if nm > 0 else np.nan})
    return pd.DataFrame(rows)
