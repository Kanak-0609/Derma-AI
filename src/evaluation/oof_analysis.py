"""oof_analysis.py - lesion-bootstrap CIs, cross-fitted calibration, referral sweep on OOF predictions."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MEL, NV = CLASS_NAMES.index('mel'), CLASS_NAMES.index('nv')


def load_oof(out_dir, tag='efficientnet_b0_weighted_ce'):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith(f'oof_{tag}_fold'))
    oof = pd.concat([pd.read_csv(f'{out_dir}/{f}') for f in files], ignore_index=True)
    y = oof.diagnosis.map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    P = oof[[f'p_{c}' for c in CLASS_NAMES]].values
    return oof, y, P


def metrics(y, P):
    pred = P.argmax(1)
    rec = recall_score(y, pred, average=None, labels=range(7), zero_division=0)
    out = {'accuracy': accuracy_score(y, pred), 'balanced_acc': rec.mean(),
           'macro_f1': f1_score(y, pred, average='macro'),
           'macro_auc': roc_auc_score(y, P, multi_class='ovr', average='macro', labels=range(7)),
           'mel_auc': roc_auc_score(y == MEL, P[:, MEL])}
    for c, r in zip(CLASS_NAMES, rec):
        out[f'recall_{c}'] = r
    return out


def lesion_bootstrap(oof, y, P, B=1000, seed=0):
    """Resamples whole lesions (not images), so repeated images of one lesion stay together."""
    rng = np.random.default_rng(seed)
    groups = list(oof.groupby('lesion_id').indices.values())
    rows = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        rows.append(metrics(y[idx], P[idx]))
    boot = pd.DataFrame(rows)
    return pd.DataFrame({'estimate': pd.Series(metrics(y, P)),
                         'ci_low': boot.quantile(0.025), 'ci_high': boot.quantile(0.975)})


def crossfit_temperature(oof, y, P, fit_temperature, apply_temperature):
    folds = oof.fold.values
    Pc, Ts = np.zeros_like(P), {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        Ts[int(k)] = fit_temperature(y[tr], P[tr])
        Pc[va] = apply_temperature(P[va], Ts[int(k)])
    return Pc, Ts


def referral_sweep(y, P, thresholds):
    pred, conf, is_mel = P.argmax(1), P.max(1), (y == MEL)
    rows = []
    for t in thresholds:
        ref = conf < t
        direct = (is_mel & (pred == MEL)).sum() / is_mel.sum()
        via = (is_mel & ref & (pred != MEL)).sum() / is_mel.sum()
        rows.append({'conf_threshold': t, 'referral_rate': ref.mean(),
                     'accuracy_on_accepted': (pred[~ref] == y[~ref]).mean() if (~ref).any() else np.nan,
                     'mel_recall_direct': direct, 'mel_recall_with_referral': direct + via,
                     'nv_share_of_referrals': (ref & (y == NV)).sum() / max(ref.sum(), 1)})
    return pd.DataFrame(rows)
