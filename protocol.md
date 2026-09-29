# Evaluation protocol (fixed before further experiments)

1. Pool: all HAM10000 images, split by lesion_id (no lesion appears on both sides of any split), stratified by diagnosis.
2. Final test set (about 10%): touched once, at the very end. No model, loss, threshold or temperature decision uses it.
3. Development data: 5-fold grouped CV (data/splits_cv/fold{k}_train.csv, fold{k}_val.csv).
4. Everything is chosen on out-of-fold validation predictions only: loss, backbone, hyperparameters, temperature, referral threshold.
5. Headline metrics carry bootstrap 95% confidence intervals, resampled by lesion.
6. Parts 5 to 9 used the earlier single split, and the earlier test set was consulted for some choices. Those results are exploratory and are not headline results.
7. Any deviation from this protocol is logged in PROGRESS.md.
