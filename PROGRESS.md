# DermaAI - Project Progress Tracker

## READ FIRST - CORRECTION TO PART 5 (supersedes conflicting statements below)
Part 5 was first written up from a single training run, which claimed Focal Loss was the "clear winner" and made efficientnet_b0_focal the reference model. A full retrain (after a Kaggle session reset, no fixed random seeds) gave different numbers. Test set, EfficientNet-B0:

| loss | run | accuracy | macro_f1 | mel recall | df recall |
|---|---|---|---|---|---|
| unweighted CE | 1 | 0.862 | 0.746 | 0.614 | 0.333 |
| unweighted CE | 2 | 0.866 | 0.746 | 0.564 | 0.476 |
| weighted CE | 1 | 0.817 | 0.712 | 0.723 | 0.524 |
| weighted CE | 2 | 0.826 | 0.777 | 0.842 | 0.714 |
| focal | 1 | 0.818 | 0.724 | 0.743 | 0.714 |
| focal | 2 | 0.806 | 0.688 | 0.792 | 0.524 |

- ROBUST (both runs): weighted CE and focal loss raise melanoma recall (~0.56-0.61 -> ~0.72-0.84) and df recall vs unweighted CE, at the cost of accuracy (~0.86 -> ~0.81-0.83).
- NOT ESTABLISHED: that focal beats weighted CE. The ordering flipped between runs. df has only 21 test images (1 image = ~5 points of recall); mel has 101.
- Part 4 claim "EfficientNet-B0 wins every metric" is too strong: in run 2 ResNet50 accuracy (0.867) ties EfficientNet-B0 (0.866); EfficientNet-B0 still leads macro-F1, precision, AUC.
- PROVISIONAL reference model for Part 6+: efficientnet_b0_weighted_ce (best or tied on average). Revisit after a multi-seed comparison.
- Run 1 results: reports/classification/imbalance_experiments/. Run 2 results: reports/classification/all_experiments/ (unweighted checkpoints are named *_ce_best.pth).
- Checkpoints are NOT in git. Backup: all_checkpoints_backup.zip (~250 MB), downloaded manually from the Kaggle Output panel.


**Purpose of this file:** paste this entire file into a new Claude conversation to resume work instantly, without needing to re-explain anything.

**Repo:** https://github.com/Kanak-0609/Derma-AI
**Compute environment:** Kaggle Notebook `notebookb67a94b9bb`, GPU: Tesla T4, connected to GitHub via a stored `GITHUB_TOKEN` Kaggle Secret, repo cloned at `/kaggle/working/repo`.
**Goal:** Build "DermaAI" - an explainable AI dermoscopic lesion analysis + clinical decision-support pipeline, for an AIIMS job application. Following a 14-part plan (data pipeline -> data quality -> segmentation -> classification -> imbalance handling -> explainability -> uncertainty -> calibration -> decision layer -> backend API -> dashboard -> experiment tracking -> model versioning -> final report/ablation).

**Kaggle -> GitHub push pattern used throughout:** create/save file in `/kaggle/working/...` -> `shutil.copy` into `/kaggle/working/repo/...` -> `git add` -> `git commit` -> `git push`.

**IMPORTANT constraint:** trained model `.pth` checkpoints (~118-125MB U-Net, smaller for classification models) exceed or risk exceeding GitHub's 100MB file limit. Decision: do NOT commit model checkpoint files to git. Keep them in Kaggle's working directory. Only commit code, metrics (CSV/JSON), and plots (PNG) to GitHub. TODO: persist checkpoints via Kaggle "Save Version" or Kaggle Models feature (not yet done).

**Known gotcha:** `src/segmentation/dataset.py` and `src/classification/dataset.py` are both named `dataset.py` (same for `transforms.py`) -- causes Python import caching collisions when both loaded in the same notebook session. Workaround: `del sys.modules['dataset']` etc. before importing the other, plus `sys.path.insert(0, ...)` to prioritize. Needs a proper fix (packages with `__init__.py`, fully-qualified imports) before Part 9, which needs both loaded together.

**Also noted:** PyTorch 2.6 changed `torch.load()` default to `weights_only=True`, which breaks loading our checkpoints (they store non-tensor metadata like per-class metrics dicts). Fix: always pass `weights_only=False` when loading our own checkpoints (safe since we trust the source).

---

## Datasets in use
- **Classification:** `kmader/skin-cancer-mnist-ham10000` (Kaggle) = HAM10000 = ISIC 2018 Task 3. 10,015 images, 600x450 RGB, 7 classes. Metadata: `HAM10000_metadata.csv` (columns: lesion_id, image_id, dx, dx_type, age, sex, localization).
- **Segmentation:** `tschandl/isic2018-challenge-task1-data-segmentation` (Kaggle) = ISIC 2018 Task 1. 2,594 training images (with masks) + 100 val / 1000 test (NO public masks -- unusable, so we made our own split from the 2,594).
- **Key fact:** these two datasets have ZERO image overlap (confirmed). Segmentation trained independently; will be used for inference-only mask generation on HAM10000 later for the Part 9 ablation study.

## Part 1 - Data pipeline: COMPLETE
- Classification: split by `lesion_id` using `StratifiedGroupKFold` (8 train/1 val/1 test folds) to avoid leakage. Result: 8009 train / 998 val / 1008 test. Zero leaked lesions confirmed.
- Segmentation: random 80/10/10 split of the 2,594 masked images. Result: 2075 train / 259 val / 260 test.
- Pushed: `data/splits/{train,validation,test,full_metadata_with_splits}.csv`, `data/splits/segmentation/{train,validation,test,full_segmentation_splits}.csv`

## Part 2 - Data Quality Module: COMPLETE
- `src/preprocessing/data_quality.py` -- corruption/dimension/mode checks, MD5 duplicate detection, imbalance calc, missing metadata, HTML report generation.
- Classification findings: 0 corrupted, uniform 600x450 RGB, 2 benign duplicate pairs (same lesion_id, same split). Class imbalance ratio: 58.3x (nv=6705 vs df=115). 57 missing ages, 57 unknown sex, 234 unknown localization.
- Segmentation findings: 0 corrupted, images vary 576x540 to 6748x4499. All 2,594 masks confirmed strictly binary (0/255) after fixing a bilinear-resize interpolation artifact by switching to `Image.NEAREST`.
- Pushed: `reports/data_quality_report.html`, `class_distribution.png`, `metadata_report.csv`, `reports/segmentation/{image,mask}_quality_report.csv`.

## Part 3 - U-Net Segmentation: COMPLETE
- `src/segmentation/unet.py` -- from-scratch U-Net, ~31M params, verified `[B,1,256,256]` output.
- `src/segmentation/dataset.py`, `transforms.py` -- masks resized with NEAREST, albumentations augmentation, ImageNet normalization.
- `src/evaluation/losses_metrics.py` -- `BCEDiceLoss`, `compute_segmentation_metrics` (Dice/IoU/Precision/Recall).
- `src/segmentation/train_segmentation.py` -- Adam lr=1e-4, ReduceLROnPlateau, checkpoints by best val_dice.
- Training: 30 epochs, batch_size=16, image_size=256. Best val_dice=0.8794 at epoch 27 (~240s/epoch).
- `src/evaluation/evaluate_segmentation.py` -- test set eval + visualization.
- FINAL TEST RESULTS (run 2 - matches reports/segmentation/): Dice=0.8922, IoU=0.8065, Precision=0.9252, Recall=0.8654 (val Dice 0.8817). Run 1 (earlier session, weights lost): Dice=0.8902, IoU=0.8031, Precision=0.9021, Recall=0.8822 (val Dice 0.8794). Segmentation is stable across runs (~0.89 Dice).
- Checkpoint `/kaggle/working/models/segmentation/unet_best.pth` NOT in git (too large).
- Pushed: all code + `reports/segmentation/sample_predictions.png`, `test_metrics.json`, `training_history.csv`.

## Part 4 - Classification (baseline CNN -> ResNet50 -> EfficientNet): COMPLETE
- Used RAW images as baseline (not segmentation-cropped) to preserve raw-vs-segmented ablation for Part 9. Input size 224x224.
- `src/classification/dataset.py` -- `ClassificationDataset`, `CLASS_NAMES = ['akiec','bcc','bkl','df','mel','nv','vasc']`, `CLASS_TO_IDX`/`IDX_TO_CLASS`.
- `src/classification/transforms.py` -- flips/rotate/affine/brightness-contrast/CoarseDropout augmentation.
- `src/classification/models.py` -- `BaselineCNN` (scratch, 391K params), ResNet50 (pretrained, 23.5M params), EfficientNet-B0 (pretrained, 4.0M params). `get_model(name)` via `MODEL_REGISTRY`.
- `src/classification/train_classification.py` -- unweighted CrossEntropyLoss (deliberate naive baseline), tracks accuracy/macro-F1/per-class P-R-F1/macro-AUC, checkpoints by best val macro-F1.
- `src/evaluation/evaluate_classification.py` -- test set eval, comparison table, confusion matrices, per-class JSON. Needed `weights_only=False` patch (see gotcha above).
- Val results (model selection): baseline_cnn 20ep -> macro-F1 0.4338. resnet50 15ep -> macro-F1 0.7509 (epoch 11, overfits after: train_f1 0.97 vs val 0.73). efficientnet_b0 15ep -> macro-F1 0.7354 (epoch 15, less overfitting: train_f1 0.93 vs val 0.74).
- FINAL TEST SET COMPARISON TABLE:
  - baseline_cnn: accuracy=0.7480, macro_precision=0.4460, macro_recall=0.3668, macro_f1=0.3899, macro_auc=0.9204
  - resnet50: accuracy=0.8552, macro_precision=0.7255, macro_recall=0.7046, macro_f1=0.6953, macro_auc=0.9654
  - efficientnet_b0 (BEST overall): accuracy=0.8621, macro_precision=0.7844, macro_recall=0.7362, macro_f1=0.7458, macro_auc=0.9752
  - EfficientNet-B0 wins every metric despite ~83% fewer params than ResNet50.
- EfficientNet-B0 per-class breakdown (precision/recall/f1/support):
  - nv: 0.916/0.948/0.932/691 | vasc: 0.933/1.000/0.966/14 | bcc: 0.786/0.805/0.795/41 | akiec: 0.645/0.769/0.702/26 | bkl: 0.813/0.684/0.743/114 | mel: 0.620/0.614/0.617/101 | df: 0.778/0.333/0.467/21
  - CRITICAL FINDINGS FOR PART 5: (1) df recall=0.333 -- smallest class (115 total images), classic imbalance failure. (2) mel (melanoma) recall=0.614 -- clinically the most serious finding in the project: best model misses ~39% of actual melanoma cases. This is the headline "before" number Part 5 must improve.
- Checkpoints at `/kaggle/working/models/classification/{baseline_cnn,resnet50,efficientnet_b0}_best.pth` -- NOT in git (size).
- Pushed: all code + `reports/classification/model_comparison_table.csv`, `per_class_metrics.json`, `*_confusion_matrix.png` (3 models).

## Part 5 - Class imbalance handling: NEXT UP (not started)
- Plan: retrain EfficientNet-B0 (best architecture from Part 4) with (a) class-weighted CrossEntropyLoss, (b) Focal Loss, as two separate experiments.
- Compare against Part 4's unweighted baseline numbers above, especially per-class recall for `mel` and `df`.
- Goal: quantify improvement, e.g. "weighted loss improved melanoma recall from 0.614 to X while accuracy changed from 0.862 to Y" -- this exact before/after comparison is a strong result for the report.

## Remaining parts (not started)
- Part 6 - Grad-CAM explainability
- Part 7 - Uncertainty estimation + abstention rule
- Part 8 - Probability calibration (temperature scaling, ECE)
- Part 9 - Decision-support layer + ablation study (raw vs preprocessed vs segmented vs segmented+augmented; needs U-Net inference on HAM10000 images -- watch for the dataset.py/transforms.py import collision gotcha above)
- Part 10 - FastAPI/Flask backend
- Part 11 - Streamlit dashboard
- Part 12 - Experiment tracking
- Part 13 - Model versioning + model_card.md (resolve checkpoint storage properly here)
- Part 14 - Final README, research report, LICENSE, Dockerfile

## User context
- GitHub username: `Kanak-0609`, repo `Derma-AI`.
- Wants the best/strongest possible model, comfortable being pushed technically.
- Wants every step fully explicit since they may hit usage limits and need to resume elsewhere.
- Treating this as an unlimited-time, from-scratch, best-effort build.


## Part 6 - Grad-CAM explainability: COMPLETE (qualitative; quantitative check deferred to Part 9)
- `src/explainability/gradcam.py`: Grad-CAM from scratch, tensor-level gradient hook on `model.features[-1]` (7x7 map). `visualize_gradcam.py`: figure generation, random samples with fixed seed 42.
- Model explained: efficientnet_b0_weighted_ce (provisional reference model, run 2).
- Test melanoma (n=101): 85 caught, 16 missed (recall 0.842). Missed were predicted as nv 7, bkl 5, bcc 2, akiec 1, df 1.
- Missed-melanoma table (`reports/explainability/mel_missed_table.csv`): mel ranked 2nd in 11/16 and 3rd in 5/16; p(mel) >= 0.30 in 5/16; 4/16 have p(mel) <= 0.07 (confident errors, e.g. nv at 0.93).
- Qualitative finding: on the rows reviewed, heat sits on the lesion, not on hair, tick marks or surrounding skin. For the one missed case read in full, p(mel)=0.46 vs bkl 0.48 with heat on the lesion -> a decision-boundary problem, not an attention problem. Only a subset of figure rows was reviewed.
- LIMITS: 7x7 map is coarse; heat on the lesion does not prove correct reasoning; one model, one run (specific image IDs change between retrains); weighted-CE probabilities are not calibrated, so p(mel) thresholds are relative scores until Part 8.
- NEXT: Part 7 (uncertainty + abstention/referral rule). Any threshold must be chosen on the VALIDATION set and reported on test, with the false-positive cost on all test images. Part 9: measure fraction of Grad-CAM heat inside U-Net lesion masks across the full test set.


## IN PROGRESS - multi-seed comparison of loss functions (decision rule set BEFORE seeing results)
- Patched train_classification.py (seed arg; `seed_everything`; seeded shuffle/workers; checkpoint stores seed) + src/evaluation/aggregate_seeds.py. Pushed as d30695f.
- Commit run: EfficientNet-B0 x {ce, weighted_ce, focal} x seeds {0,1,2}, 15 epochs each, outputs to reports/classification/multiseed/{per_run.csv, summary.csv}.
- Decision rule: report mean +/- std over seeds for accuracy, macro-F1, mel recall, df recall. If a difference between losses is smaller than the seed-to-seed spread, call them INDISTINGUISHABLE (n=3 spread is only a rough estimate). Tie-break on a stated secondary criterion: weighted CE (no extra hyperparameter) over focal (adds gamma).
- Seeding limits: GPU kernels and albumentations' own RNG are not controlled, so runs are independent replicates, not bit-reproducible.
- U-Net weights (unet_best.pth) and ResNet50/baseline checkpoints exist only in all_checkpoints_backup.zip (Kaggle Output panel / GitHub release `checkpoints-run2` if the upload succeeded). The seed run does NOT regenerate them.


## Part 5 multi-seed comparison: COMPLETE - DECISION MADE
3 seeds x {ce, weighted_ce, focal}, EfficientNet-B0, 15 epochs each. Full results: reports/classification/multiseed/{per_run,summary}.csv.

ROBUST (ranges don't overlap across all 3 seeds of each loss):
- Melanoma recall: ce [0.653-0.733] vs weighted_ce [0.762-0.812] vs focal [0.752-0.782]. Every seed of both imbalance-aware losses beats every seed of plain CE. This is the real, reproducible finding of Part 5.
- Accuracy: ce [0.852-0.864] vs weighted_ce [0.819-0.823] vs focal [0.791-0.838, 2/3 seeds below ce's min]. Confirms a genuine accuracy cost for imbalance handling.

NOT DISTINGUISHABLE (ranges overlap, n=3 seeds):
- weighted_ce vs focal on mel recall (0.789 vs 0.769, overlapping ranges) - cannot claim either is better.
- df recall across all 3 losses (std 0.05-0.10 on a metric ~0.52-0.57; only 21 test images, ~5 points per flipped prediction).
- macro-F1 across all 3 losses (ce actually has highest mean 0.751, but overlaps focal's range).

DECISION (pre-registered tie-break rule applied): since weighted_ce and focal are statistically indistinguishable on melanoma recall (the metric that matters), weighted_ce is the reference model going forward - no extra hyperparameter (gamma) to justify, same performance.

REFERENCE MODEL FOR PART 6 ONWARD: efficientnet_b0_weighted_ce (this replaces the earlier single-run focal-loss pick, which the multi-seed data does not support as a distinct winner).
Part 6 (Grad-CAM) was already run against efficientnet_b0_weighted_ce_best.pth - no rework needed there.


## Part 7 referral rule: COMPLETE
Rule: refer when top-class confidence < 0.8 (mel-specific trigger disabled). Chosen on validation only, applied once to test.
Model: efficientnet_b0_weighted_ce (release checkpoint).
Validation: mel recall 0.723 -> 0.807 at 24.6% referral.
Test: mel recall 0.842 -> 0.980 at 26.1% referral (263 images). Referral counts a flagged miss as caught.
Note: test baseline mel recall (0.842) is higher than validation (0.723), so the absolute gain differs.
Referral rate by true class (test): akiec 0.308, bcc 0.195, bkl 0.377, df 0.333, mel 0.347, nv 0.233, vasc 0.071.
Files: reports/referral/test_result.csv, test_per_sample.csv, src/evaluation/referral.py


## Part 8 calibration: COMPLETE
Temperature scaling, T=1.510 fitted on validation only (T>1: model was overconfident).
ECE validation 0.0965 -> 0.0490 (optimistic, T fitted here). ECE test 0.0504 -> 0.0371, NLL test 0.4525 -> 0.4512 (modest gain).
Argmax unchanged, so accuracy identical.
Reusing 0.8 on calibrated probs raised referral to 40.3% val / 40.7% test, so threshold was re-picked on validation.
Calibrated rule: conf < 0.65 -> val referral 23.7%, mel recall with referral 0.807 (same as uncalibrated 0.8 rule at 24.6%).
Test was already used once for the uncalibrated rule; the calibrated result is reported as a variant, not a fresh untouched test.
Files: src/evaluation/calibration.py, reports/calibration/


## Part 9 decision-support layer: CORE COMPLETE
src/decision_support.py: DecisionSupport class. Image -> predicted class, calibrated probabilities (T from reports/calibration/temperature.json), referral flag (conf < 0.65), plain-language message with disclaimer.
Verified on all 1008 test images: max abs diff vs earlier calibrated probs = 0.00000, referred = 256 (matches Part 8).
Removed an unvalidated p(mel) >= 0.10 message warning; message now uses only the locked referral rule plus a note when melanoma is the top prediction.
test.csv is sorted by class (first rows all bkl), so never judge rates from a small head sample.
Mask (U-Net) and heatmap (Grad-CAM) hooks exist as optional callables, not yet wired in.


## Part 9 decision-support layer: COMPLETE (mask + heatmap wired in)
src/explain_wrappers.py: make_mask_fn (U-Net, 256px, sigmoid on logits, threshold 0.5) and make_cam_fn (hand-written Grad-CAM on model.features[-1]).
Checked on one test image per class (reports/explain/decision_support_demo.png): masks follow the lesion on bkl/df/mel/vasc; loose on bcc; poor on nv (close-up, no visible border, coverage 0.61) and akiec (diffuse edges).
Grad-CAM is centred on the lesion in most rows; nearly always centred, so treat as qualitative only.
U-Net raw output is logits (range about -7 to +9), so sigmoid is correct.
7-image visual check only; the quantitative segmentation result is the Part 3 Dice/IoU.


## Phase A step 1: evaluation protocol and grouped splits - DONE
protocol.md fixes the rules: lesion-grouped, class-stratified splits; a final test set touched once at the end; all choices made on out-of-fold validation only.
Splits: data/splits_cv/ (fold0-4 train/val, final_test.csv, all_splits.csv), script src/data/make_cv_splits.py, seed 42.
Old-split check: 0 of 7470 lesions shared across the earlier train/validation/test files, so the Part 5-9 results had no lesion leakage. They remain exploratory because the old test set was used for some decisions.
Dev 9023 images (5 folds), final test 992 (df only 10, vasc 15, so report wide CIs for those classes).


## Phase A step 2-3: 5-fold CV baseline and operating point - DONE
EfficientNet-B0, weighted CE, 15 epochs, final epoch used, cosine LR (src/classification/train_cv.py). Pooled OOF over 9023 images:
accuracy 0.810 (0.801-0.820), macro-F1 0.721 (0.700-0.740), macro-AUC 0.959 (0.952-0.965), melanoma recall 0.695 (0.661-0.728), melanoma AUC 0.914.
Wide CIs for df 0.637-0.835, akiec 0.598-0.739, vasc 0.823-0.951.
Cross-fitted temperature per fold 1.18-1.21; ECE 0.0408 -> 0.0102.
Locked operating point: calibrated confidence < 0.70 (26.7% referral). Part 5-9 numbers (melanoma recall about 0.79) were optimistic because of best-epoch selection.


## Final test set: OPENED ONCE (5-fold ConvNeXt-Tiny ensemble, 992 images) - now spent
Test: accuracy 0.889 (0.863-0.912), macro-F1 0.854 (0.793-0.897), macro-AUC 0.986, melanoma AUC 0.964, melanoma recall 0.829 (0.732-0.911).
Single-model out-of-fold reference: accuracy 0.846, macro-F1 0.766, melanoma recall 0.722. The test numbers are higher: ensemble effect plus one favourable 992-image draw; df 10/10 and vasc 15 images are tiny. Quote both, treat out-of-fold as the conservative estimate.
Referral (T=1.419, threshold 0.79, both fitted on single-model out-of-fold predictions): 29.9% referral vs 26.7% target; accuracy on accepted 0.973; melanoma recall direct 0.829, with referral 0.973 (a flagged miss counts as caught).
Calibration got WORSE on the test set: ECE 0.029 raw -> 0.054 after temperature. Cause: T was fitted on single-model predictions but applied to a 5-model average, which is already softer. Reported as found; T and threshold were not re-tuned on the test set.


## Phase C: Grad-CAM and conformal analysis (ConvNeXt-T, fold-0 model, 150 val images)
Localisation: 34.8% of Grad-CAM energy falls inside the U-Net mask (mask covers 24%), ratio 1.45x chance; peak inside the mask 54% (chance 24%). Coarse 7x7 maps, imperfect masks: qualitative aid only.
Randomisation check (final layer only): Spearman 0.139 between real and randomised-head maps.
The first deletion test compared against per-pixel random deletion, which is speckle noise and confounds the comparison; replaced by block-level deletion (reports/analysis/gradcam_deletion_block.csv).
Conformal (class-conditional, calibrated across folds): coverage 0.949 / 0.900 / 0.800 at alpha 0.05 / 0.10 / 0.20, min class coverage 0.937 / 0.866 / 0.762.
At alpha 0.10: referral 24.7%, accuracy on single-class sets 0.909, melanoma recall with referral 0.919. A confidence rule at the same referral rate gives 0.937 accuracy on accepted and 0.902 melanoma recall with referral, so conformal buys a coverage guarantee, not a better trade-off.
