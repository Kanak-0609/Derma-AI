"""
referral.py - DermaAI Uncertainty Estimation + Referral Rule

Two independent referral triggers:
    1. low_confidence: top predicted probability below a threshold (general uncertainty)
    2. mel_suspicion: p(mel) at or above a threshold, regardless of the predicted class
       (catches cases where melanoma is a strong runner-up, not just the top pick)

Thresholds are chosen on the VALIDATION set only, then applied once to the TEST set
for reporting. This module never lets a threshold be fit on test-set outcomes.
"""

import numpy as np
import pandas as pd


def evaluate_referral_thresholds(labels, probs, class_names, mel_thresholds, conf_thresholds):
    mel_idx = class_names.index('mel')
    preds = probs.argmax(axis=1)
    top_prob = probs.max(axis=1)
    p_mel = probs[:, mel_idx]

    results = []
    for mel_t in mel_thresholds:
        for conf_t in conf_thresholds:
            referred = (top_prob < conf_t) | (p_mel >= mel_t)

            is_mel = labels == mel_idx
            caught_directly = is_mel & (preds == mel_idx)
            caught_via_referral = is_mel & (preds != mel_idx) & referred
            mel_recall_with_referral = (caught_directly | caught_via_referral).sum() / max(is_mel.sum(), 1)
            mel_recall_without_referral = caught_directly.sum() / max(is_mel.sum(), 1)

            results.append({
                'mel_threshold': mel_t,
                'conf_threshold': conf_t,
                'mel_recall_without_referral': mel_recall_without_referral,
                'mel_recall_with_referral': mel_recall_with_referral,
                'referral_rate': referred.mean(),
                'n_referred': int(referred.sum()),
            })
    return pd.DataFrame(results)


def apply_referral(labels, probs, class_names, mel_threshold, conf_threshold):
    mel_idx = class_names.index('mel')
    preds = probs.argmax(axis=1)
    top_prob = probs.max(axis=1)
    p_mel = probs[:, mel_idx]
    referred = (top_prob < conf_threshold) | (p_mel >= mel_threshold)

    return pd.DataFrame({
        'true_label': [class_names[i] for i in labels],
        'predicted': [class_names[i] for i in preds],
        'top_prob': top_prob,
        'p_mel': p_mel,
        'referred_for_review': referred,
    })
