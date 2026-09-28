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
