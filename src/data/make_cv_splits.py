"""
make_cv_splits.py - lesion-grouped, class-stratified splits (Phase A).

Outputs in --out_dir:
    final_test.csv                      locked test set, touched once at the end
    fold{k}_train.csv, fold{k}_val.csv  5-fold CV on the remaining development data
    all_splits.csv                      every image with its role and fold
"""
import argparse, os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_pool(split_dir):
    parts = []
    for name in ('train', 'validation', 'test'):
        d = pd.read_csv(f'{split_dir}/{name}.csv')
        d['old_split'] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main(split_dir, out_dir, seed=42, test_frac_folds=10, n_folds=5):
    os.makedirs(out_dir, exist_ok=True)
    pool = load_pool(split_dir)
    print(f"Rows loaded: {len(pool)} | duplicate image_id: {pool.image_id.duplicated().sum()}")
    pool = pool.drop_duplicates('image_id').reset_index(drop=True)

    # 1) leakage check on the OLD splits
    spread = pool.groupby('lesion_id')['old_split'].nunique()
    print(f"Old splits: {int((spread > 1).sum())} of {pool.lesion_id.nunique()} lesions appear in more than one split")

    y, g = pool['diagnosis'], pool['lesion_id']

    # 2) locked final test set: one fold of a 10-fold grouped split
    outer = StratifiedGroupKFold(n_splits=test_frac_folds, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(pool, y, g))
    test = pool.iloc[test_idx].copy()
    dev = pool.iloc[dev_idx].reset_index(drop=True)

    # 3) 5-fold grouped CV on the development data
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dev['fold'] = -1
    for k, (_, val_idx) in enumerate(inner.split(dev, dev['diagnosis'], dev['lesion_id'])):
        dev.loc[val_idx, 'fold'] = k

    # 4) verify: no lesion overlaps
    assert set(test.lesion_id).isdisjoint(set(dev.lesion_id)), "lesion leak: test vs dev"
    for k in range(n_folds):
        tr, va = dev[dev.fold != k], dev[dev.fold == k]
        assert set(tr.lesion_id).isdisjoint(set(va.lesion_id)), f"lesion leak in fold {k}"
        tr.to_csv(f'{out_dir}/fold{k}_train.csv', index=False)
        va.to_csv(f'{out_dir}/fold{k}_val.csv', index=False)
    test.to_csv(f'{out_dir}/final_test.csv', index=False)

    dev['role'] = 'dev'
    test['role'], test['fold'] = 'final_test', -1
    pd.concat([dev, test]).to_csv(f'{out_dir}/all_splits.csv', index=False)

    # 5) class counts
    counts = {f'fold{k}_val': dev[dev.fold == k].diagnosis.value_counts() for k in range(n_folds)}
    counts['final_test'] = test.diagnosis.value_counts()
    print("\nImages per class:")
    print(pd.DataFrame(counts).fillna(0).astype(int).to_string())
    print(f"\nDev images: {len(dev)} | final test images: {len(test)} | all leakage checks passed")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_dir', default='/kaggle/working/repo/data/splits')
    ap.add_argument('--out_dir', default='/kaggle/working/repo/data/splits_cv')
    a = ap.parse_args()
    main(a.split_dir, a.out_dir)
