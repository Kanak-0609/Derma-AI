"""
calibration.py - DermaAI probability calibration

- ece(): Expected Calibration Error on top-label confidence
- fit_temperature(): finds a single temperature T on VALIDATION only
- apply_temperature(): rescales probabilities with T

Temperature scaling on log(probs) is equivalent to scaling the logits,
because softmax ignores a constant shift. Argmax never changes, so accuracy
is identical before and after; only the confidence values move.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(probs, T):
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    return _softmax(logp / T)


def nll(labels, probs):
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(val_labels, val_probs):
    res = minimize_scalar(lambda T: nll(val_labels, apply_temperature(val_probs, T)),
                          bounds=(0.05, 10.0), method='bounded')
    return float(res.x)


def ece(labels, probs, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        total += m.mean() * abs(acc - c)
        rows.append((c, acc, int(m.sum())))
    return float(total), rows


def reliability_plot(rows_before, rows_after, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='perfect')
    for rows, name in ((rows_before, 'before'), (rows_after, 'after')):
        ax.plot([r[0] for r in rows], [r[1] for r in rows], 'o-', label=name)
    ax.set_xlabel('mean confidence')
    ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
