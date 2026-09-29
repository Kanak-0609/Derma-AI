"""external_validation.py - evaluate on ISIC 2019 images that are NOT in HAM10000."""
import os
import numpy as np, pandas as pd
from common import CLASS_NAMES, MEL, predict_paths, clf_val_transform
from oof_analysis import metrics
from calibration import apply_temperature, ece

MAP = {'MEL': 'mel', 'NV': 'nv', 'BCC': 'bcc', 'AK': 'akiec', 'BKL': 'bkl', 'DF': 'df', 'VASC': 'vasc'}


def find_isic2019(root='/kaggle/input'):
    gt, imgs = None, {}
    for d, _, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if fl == 'isic_2019_training_groundtruth.csv':
                gt = os.path.join(d, f)
            elif fl.endswith('.jpg') and f.startswith('ISIC_'):
                imgs[f[:-4]] = os.path.join(d, f)
    return gt, imgs


def build_external(ham_ids, max_images=6000, seed=0):
    gt, imgs = find_isic2019()
    if gt is None:
        raise FileNotFoundError('ISIC_2019_Training_GroundTruth.csv not found under /kaggle/input. '
                                'Use "Add Data" in the notebook to attach an ISIC 2019 training dataset.')
    df = pd.read_csv(gt)
    cols = [c for c in MAP if c in df.columns]
    df = df[df[cols].sum(1) > 0].copy()                # drops SCC-only and UNK rows (no HAM10000 equivalent)
    df['label'] = df[cols].idxmax(1).map(MAP)
    n0 = len(df)
    df = df[~df.image.isin(ham_ids)].copy()            # exclude every image that is in HAM10000
    df['image_path'] = df.image.map(imgs)
    df = df.dropna(subset=['image_path'])
    print(f'ISIC 2019 mapped rows: {n0} | after removing HAM10000 ids and missing files: {len(df)}')
    if len(df) > max_images:
        df = df.sample(max_images, random_state=seed)
    print(df.label.value_counts().to_string())
    return df.reset_index(drop=True)


def run_external(ens, T, thr, ham_ids, size, device, out_dir, oof_metrics, max_images=6000, B=500, seed=0):
    df = build_external(ham_ids, max_images)
    y = df.label.map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    P = predict_paths(ens, df.image_path.tolist(), clf_val_transform(size), device)
    Pc = apply_temperature(P, T)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(B):                                  # image-level bootstrap (lesion ids unavailable)
        idx = rng.integers(0, len(y), len(y))
        try:
            rows.append(metrics(y[idx], P[idx]))
        except ValueError:
            continue
    b = pd.DataFrame(rows)
    tab = pd.DataFrame({'external': pd.Series(metrics(y, P)), 'ci_low': b.quantile(0.025),
                        'ci_high': b.quantile(0.975), 'internal_oof': pd.Series(oof_metrics)})
    print('\n=== EXTERNAL (ISIC 2019, HAM10000 removed) vs internal out-of-fold ===')
    print(tab.round(3).to_string())
    pred, ref, is_mel = P.argmax(1), Pc.max(1) < thr, y == MEL
    res = {'n': int(len(y)), 'referral_rate': float(ref.mean()),
           'accuracy_on_accepted': float((pred[~ref] == y[~ref]).mean()),
           'mel_recall_direct': float((pred[is_mel] == MEL).mean()),
           'mel_recall_with_referral': float(((pred == MEL) | ref)[is_mel].mean()),
           'ece_raw': ece(y, P)[0], 'ece_calibrated': ece(y, Pc)[0]}
    print('\nreferral and calibration under shift:', res)
    os.makedirs(out_dir, exist_ok=True)
    tab.round(4).to_csv(f'{out_dir}/external_metrics.csv')
    pd.Series(res).to_csv(f'{out_dir}/external_referral.csv')
    return tab, res
