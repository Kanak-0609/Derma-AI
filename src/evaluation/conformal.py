"""conformal.py - class-conditional (Mondrian) split-conformal prediction, evaluated across the CV folds."""
import numpy as np, pandas as pd
from common import MEL


def class_thresholds(y, P, alpha):
    q = {}
    for c in range(P.shape[1]):
        s = np.sort(1 - P[y == c, c])
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[c] = np.inf if k > n else s[k - 1]
    return q


def prediction_sets(P, q):
    return np.stack([(1 - P[:, c]) <= q[c] for c in range(P.shape[1])], 1)


def crossfold_conformal(oof, y, P, alphas=(0.05, 0.10, 0.20)):
    folds = oof.fold.values
    pred, conf, is_mel = P.argmax(1), P.max(1), (y == MEL)
    rows = []
    for a in alphas:
        S = np.zeros_like(P, dtype=bool)
        for k in sorted(set(folds)):
            tr, va = folds != k, folds == k
            S[va] = prediction_sets(P[va], class_thresholds(y[tr], P[tr], a))   # calibrated on the other folds
        size = S.sum(1)
        single = size == 1
        cover = S[np.arange(len(y)), y]
        referred = ~single                                                        # empty or multi-class sets
        ref_c = conf < np.quantile(conf, referred.mean())                         # confidence rule, same referral rate
        rows.append({'alpha': a, 'coverage': cover.mean(), 'coverage_mel': cover[is_mel].mean(),
                     'min_class_coverage': min(cover[y == c].mean() for c in range(P.shape[1])),
                     'mean_set_size': size.mean(), 'referral_rate': referred.mean(),
                     'acc_on_singletons': (pred[single] == y[single]).mean(),
                     'mel_recall_direct': (pred[is_mel] == MEL).mean(),
                     'mel_recall_with_referral': ((pred == MEL) | referred)[is_mel].mean(),
                     'conf_rule_mel_recall_with_referral': ((pred == MEL) | ref_c)[is_mel].mean(),
                     'conf_rule_acc_on_accepted': (pred[~ref_c] == y[~ref_c]).mean()})
    return pd.DataFrame(rows)
