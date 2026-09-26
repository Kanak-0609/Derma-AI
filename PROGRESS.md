# DermaAI - Project Progress Tracker

**Purpose of this file:** paste this entire file into a new Claude conversation to resume work instantly, without needing to re-explain anything.

**Repo:** https://github.com/Kanak-0609/Derma-AI
**Compute environment:** Kaggle Notebook `notebookb67a94b9bb`, GPU: Tesla T4, connected to GitHub via a stored `GITHUB_TOKEN` Kaggle Secret, repo cloned at `/kaggle/working/repo`.
**Goal:** Build "DermaAI" - an explainable AI dermoscopic lesion analysis + clinical decision-support pipeline, for an AIIMS job application (deadline 30 Sept 2026). Following a 14-part plan (data pipeline -> data quality -> segmentation -> classification -> imbalance handling -> explainability -> uncertainty -> calibration -> decision layer -> backend API -> dashboard -> experiment tracking -> model versioning -> final report/ablation).

**Kaggle -> GitHub push pattern used throughout:** create/save file in `/kaggle/working/...` -> `shutil.copy` into `/kaggle/working/repo/...` -> `git add` -> `git commit` -> `git push`.

**IMPORTANT constraint discovered:** trained model `.pth` checkpoints (~118-125MB for the U-Net) exceed GitHub's 100MB file limit. Decision: do NOT commit model checkpoint files to git. Keep them in Kaggle's working directory / Kaggle Models feature. Only commit code, metrics (CSV/JSON), and plots (PNG) to GitHub. Revisit with Git LFS in Part 13 if needed.

---

## Datasets in use
- **Classification:** `kmader/skin-cancer-mnist-ham10000` (Kaggle) = HAM10000 = ISIC 2018 Task 3. 10,015 images, 600x450 RGB, 7 classes. Metadata: `HAM10000_metadata.csv` (columns: lesion_id, image_id, dx, dx_type, age, sex, localization).
- **Segmentation:** `tschandl/isic2018-challenge-task1-data-segmentation` (Kaggle) = ISIC 2018 Task 1. 2,594 training images (with masks) + 100 val / 1000 test (NO public masks -- unusable for our evaluation, so we made our own split from the 2,594).
- **Key fact:** these two datasets have ZERO image overlap (confirmed) -- they are disjoint pools from the original 2018 challenge. Architecture handles this as two separate sub-pipelines (segmentation trained independently; will be used for inference-only mask generation on HAM10000 later for the Part 9 ablation study).

## Part 1 - Data pipeline: COMPLETE
- Classification: split by `lesion_id` using `StratifiedGroupKFold` (10 folds: 8 train / 1 val / 1 test) to avoid patient/lesion leakage. Result: 8009 train / 998 val / 1008 test. Zero leaked lesions confirmed.
- Segmentation: random 80/10/10 split of the 2,594 masked images. Result: 2075 train / 259 val / 260 test.
- Files pushed to repo:
  - `data/splits/train.csv`, `validation.csv`, `test.csv`, `full_metadata_with_splits.csv` (classification)
  - `data/splits/segmentation/train.csv`, `validation.csv`, `test.csv`, `full_segmentation_splits.csv`

## Part 2 - Data Quality Module: COMPLETE
- Built `src/preprocessing/data_quality.py` -- reusable module: image corruption/dimension/mode checks, duplicate detection via MD5 hash, class imbalance calc, missing metadata, HTML report generation.
- **Classification dataset findings:** 0 corrupted images, all uniform 600x450 RGB, 2 duplicate file pairs found (both benign). **Class imbalance ratio: 58.3x** (nv=6705 vs df=115). 57 missing ages, 57 "unknown" sex, 234 "unknown" localization.
- **Segmentation dataset findings:** 0 corrupted images. Images vary wildly in size (576x540 to 6748x4499). Masks: confirmed all 2,594 strictly binary (0/255) -- note: initial fast-check using bilinear resize falsely flagged all masks as non-binary due to interpolation artifacts; switching to `Image.NEAREST` resampling fixed this.
- Reports pushed: `reports/data_quality_report.html`, `class_distribution.png`, `metadata_report.csv`, `reports/segmentation/image_quality_report.csv`, `mask_quality_report.csv`.

## Part 3 - U-Net Segmentation: COMPLETE
- Built from scratch: `src/segmentation/unet.py` (encoder-decoder, skip connections, ~31M params, verified output shape `[B,1,256,256]` on GPU).
- `src/segmentation/dataset.py` -- `SegmentationDataset` class, masks resized with NEAREST.
- `src/segmentation/transforms.py` -- albumentations pipelines (train: flips/rotate/brightness-contrast; val: resize+normalize only). ImageNet mean/std normalization.
- `src/evaluation/losses_metrics.py` -- `BCEDiceLoss` (combined BCE + Dice, 50/50) + `compute_segmentation_metrics()` (Dice, IoU, Precision, Recall).
- `src/segmentation/train_segmentation.py` -- full training loop, Adam lr=1e-4, ReduceLROnPlateau, saves best checkpoint by val_dice, logs history to CSV.
- **Training result (30 epochs, batch_size=16, image_size=256):** Best val_dice = 0.8794 at epoch 27. ~240s/epoch.
- `src/evaluation/evaluate_segmentation.py` -- loads checkpoint, evaluates on test set, generates side-by-side visualization.
- **FINAL TEST SET RESULTS: Dice=0.8902, IoU=0.8031, Precision=0.9021, Recall=0.8822.**
- Checkpoint location: `/kaggle/working/models/segmentation/unet_best.pth` (NOT in git -- too large). TODO: persist via Kaggle Save Version or Kaggle Models.
- Pushed to repo: all code above + `reports/segmentation/sample_predictions.png`, `test_metrics.json`, `training_history.csv`.

## Part 4 - Classification (baseline CNN -> ResNet50 -> EfficientNet): IN PROGRESS
- Decision: start with RAW images as the true baseline (not segmentation-cropped), to preserve the "raw vs segmented" ablation comparison for Part 9.
- Input size for classification: 224x224 (ImageNet-pretrained backbone standard), different from segmentation's 256x256.
- **Done so far:** `src/classification/dataset.py` written -- `ClassificationDataset` class with fixed `CLASS_NAMES = ['akiec','bcc','bkl','df','mel','nv','vasc']` and `CLASS_TO_IDX`/`IDX_TO_CLASS` mappings. NOT yet pushed to git.
- **Not yet done:** classification augmentation transforms, baseline CNN model, ResNet50, EfficientNet, training loop, evaluation, comparison table (Model | Accuracy | Precision | Recall | F1 | ROC-AUC).

## Remaining parts (not started)
- Part 5 - Class imbalance handling (weighted loss vs focal loss, compare)
- Part 6 - Grad-CAM explainability
- Part 7 - Uncertainty estimation + abstention rule
- Part 8 - Probability calibration (temperature scaling, ECE)
- Part 9 - Decision-support layer + ablation study
- Part 10 - FastAPI/Flask backend
- Part 11 - Streamlit dashboard
- Part 12 - Experiment tracking
- Part 13 - Model versioning + model_card.md
- Part 14 - Final README, research report, LICENSE, Dockerfile

## User context
- GitHub username: `Kanak-0609`, repo `Derma-AI`.
- Wants the best/strongest possible model, comfortable being pushed technically.
- Wants every step fully explicit since they may hit usage limits and need to resume elsewhere.
- Treating this as an unlimited-time, from-scratch, best-effort build.
