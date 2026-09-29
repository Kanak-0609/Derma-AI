"""final_eval.py - derive the operating point on OOF data, then open the final test set ONCE."""
import os, sys, json
import numpy as np, pandas as pd, torch
from common import ROOT, CKPT, CLASS_NAMES, MEL, predict_paths, clf_val_transform, ensure_checkpoints
from backbones import load_member, EnsembleLogProb
from oof_analysis import load_oof, metrics
from calibration import fit_temperature, apply_temperature, ece

TARGET_REFERRAL = 0.267     # the referral rate of the locked baseline operating point (protocol.md item 8)
OUT = f'{ROOT}/reports/final'


def choose_threshold(y, Pc, target):
    grid = np.round(np.arange(0.40, 0.951, 0.01), 2)
    rates = np.array([(Pc.max(1) < t).mean() for t in grid])
    return float(grid[np.argmin(np.abs(rates - target))])


def boot(df, y, P, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = list(df.groupby('lesion_id').indices.values())
    rows = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            rows.append(metrics(y[idx], P[idx]))
        except ValueError:
            continue
    b = pd.DataFrame(rows)
    return pd.DataFrame({'estimate': pd.Series(metrics(y, P)), 'ci_low': b.quantile(0.025), 'ci_high': b.quantile(0.975)})


def run(confirm_open_final_test=False):
    if not confirm_open_final_test:
        raise RuntimeError('The final test set is locked (protocol.md). Set CONFIRM = True only when the model, '
                           'temperature and threshold are final, and do it once.')
    choice = json.load(open(f'{ROOT}/reports/cv/final_choice.json'))
    tag, size = choice['tag'], choice['image_size']
    ensure_checkpoints(tag)
    oof, y, P = load_oof(f'{ROOT}/reports/cv', tag)
    T = fit_temperature(y, P)
    thr = choose_threshold(y, apply_temperature(P, T), TARGET_REFERRAL)
    oofm = metrics(y, P)
    os.makedirs(OUT, exist_ok=True)
    op = {'tag': tag, 'model_name': choice['model_name'], 'image_size': size, 'temperature': T,
          'calibrated_conf_threshold': thr, 'target_referral_rate': TARGET_REFERRAL,
          'oof_referral_rate_at_threshold': float((apply_temperature(P, T).max(1) < thr).mean()),
          'chosen_on': 'pooled out-of-fold predictions'}
    json.dump(op, open(f'{OUT}/operating_point.json', 'w'), indent=2)
    print('OPERATING POINT (from OOF only):', op)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ens = EnsembleLogProb([load_member(f'{CKPT}/{tag}_fold{k}_last.pth', device) for k in range(5)]).eval()
    test = pd.read_csv(f'{ROOT}/data/splits_cv/final_test.csv')
    yt = test.diagnosis.map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    Pt = predict_paths(ens, test.image_path.tolist(), clf_val_transform(size), device)
    Ptc = apply_temperature(Pt, T)

    ci = boot(test, yt, Pt)
    ci['oof_reference'] = pd.Series(oofm)
    print('\n=== FINAL TEST (5-fold ensemble), 95% lesion-bootstrap CI vs pooled OOF ===')
    print(ci.round(3).to_string())
    ci.round(4).to_csv(f'{OUT}/final_test_metrics_ci.csv')

    pred, ref, is_mel = Pt.argmax(1), Ptc.max(1) < thr, yt == MEL
    res = {'referral_rate': float(ref.mean()), 'accuracy_all': float((pred == yt).mean()),
           'accuracy_on_accepted': float((pred[~ref] == yt[~ref]).mean()),
           'mel_recall_direct': float((pred[is_mel] == MEL).mean()),
           'mel_recall_with_referral': float(((pred == MEL) | ref)[is_mel].mean()),
           'nv_share_of_referrals': float((ref & (yt == CLASS_NAMES.index('nv'))).sum() / max(ref.sum(), 1)),
           'ece_raw': ece(yt, Pt)[0], 'ece_calibrated': ece(yt, Ptc)[0],
           'temperature': T, 'threshold': thr, 'n_test': int(len(test))}
    print('\n=== referral on final test ===')
    print(json.dumps(res, indent=2))
    print('\nreferral rate by true class:')
    print(pd.Series(ref).groupby(test.diagnosis.values).mean().round(3).to_string())
    json.dump(res, open(f'{OUT}/final_test_referral.json', 'w'), indent=2)
    pdf = test[['image_id', 'lesion_id', 'diagnosis']].copy()
    for i, c in enumerate(CLASS_NAMES):
        pdf[f'p_{c}'] = Pt[:, i]
    pdf.to_csv(f'{OUT}/final_test_predictions.csv', index=False)
