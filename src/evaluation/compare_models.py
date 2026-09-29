"""compare_models.py - paired lesion-bootstrap comparison of two models' out-of-fold predictions."""
import numpy as np, pandas as pd
from oof_analysis import metrics

KEYS = ('macro_f1', 'balanced_acc', 'macro_auc', 'mel_auc', 'recall_mel')


def paired_bootstrap(oof, y, P_base, P_new, B=500, seed=0):
    rng = np.random.default_rng(seed)
    groups = list(oof.groupby('lesion_id').indices.values())
    rows = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        a, b = metrics(y[idx], P_base[idx]), metrics(y[idx], P_new[idx])
        rows.append({k: b[k] - a[k] for k in KEYS})
    d = pd.DataFrame(rows)
    ma, mb = metrics(y, P_base), metrics(y, P_new)
    return pd.DataFrame({'diff': {k: mb[k] - ma[k] for k in KEYS},
                         'ci_low': d.quantile(0.025), 'ci_high': d.quantile(0.975)})
