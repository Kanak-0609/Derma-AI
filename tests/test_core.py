import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'evaluation'))
from calibration import apply_temperature, ece, fit_temperature
from referral import apply_referral
from oof_analysis import referral_sweep

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _sample(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 7)) * 2
    p = _softmax(z)
    y = np.array([rng.choice(7, p=pi) for pi in p])
    return z, p, y


def test_temperature_one_is_identity():
    _, p, _ = _sample(500)
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-6)


@pytest.mark.parametrize('T', [0.5, 2.0, 5.0])
def test_temperature_keeps_argmax(T):
    _, p, _ = _sample(500)
    assert (apply_temperature(p, T).argmax(1) == p.argmax(1)).all()


def test_temperature_above_one_lowers_confidence():
    _, p, _ = _sample(500)
    assert apply_temperature(p, 2.0).max(1).mean() < p.max(1).mean()


def test_fit_temperature_detects_overconfidence():
    z, _, y = _sample()
    T = fit_temperature(y, _softmax(3 * z))
    assert 2.0 < T < 4.5


def test_ece_small_when_calibrated_and_larger_when_overconfident():
    z, p, y = _sample()
    assert ece(y, p)[0] < 0.06
    assert ece(y, _softmax(3 * z))[0] > ece(y, p)[0]


def test_referral_rate_increases_with_threshold():
    _, p, y = _sample(1000)
    rates = referral_sweep(y, p, [0.4, 0.5, 0.6, 0.7, 0.8]).referral_rate.values
    assert (np.diff(rates) >= 0).all()


def test_referral_flags_low_confidence_only():
    probs = np.array([[0.05, 0.05, 0.05, 0.05, 0.05, 0.65, 0.10],
                      [0.02, 0.02, 0.02, 0.02, 0.02, 0.88, 0.02]])
    out = apply_referral(np.array([5, 5]), probs, CLASSES, 1.1, 0.70)
    assert out.referred_for_review.tolist() == [True, False]


def test_cv_folds_have_no_lesion_leak():
    d = ROOT / 'data' / 'splits_cv'
    if not (d / 'final_test.csv').exists():
        pytest.skip('splits not present')
    test = set(pd.read_csv(d / 'final_test.csv').lesion_id)
    for k in range(5):
        tr, va = pd.read_csv(d / f'fold{k}_train.csv'), pd.read_csv(d / f'fold{k}_val.csv')
        assert set(tr.lesion_id).isdisjoint(set(va.lesion_id))
        assert test.isdisjoint(set(tr.lesion_id) | set(va.lesion_id))
