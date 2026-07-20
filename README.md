# Gait Disease Detection — MHI + CNN (Healthy / KOA / PD)

A modified reimplementation of *"Vision-based approach to knee osteoarthritis
and Parkinson's disease detection utilizing human gait patterns"*
(Ali et al., PeerJ Computer Science, 2025), replacing the base paper's
**Mask R-CNN silhouette segmentation + CNN + GRU** pipeline with a
**Motion History Image (MHI) + CNN** pipeline (no silhouettes, no GRU).

```
Videos → Motion History Image (one MHI per video) → resize 224×224
       → Transfer-learning CNN (VGG16 / MobileNet / DenseNet)
       → Fully Connected layers → Softmax → Healthy / KOA / PD
```

## Why no GRU?

The base paper's GRU models temporal dependencies **across frames of one
video**. Collapsing a whole video into a single MHI already encodes that
temporal information (via the recursive intensity-decay rule — see
`src/preprocessing/mhi_generator.py`), so there is no sequence dimension
left for a GRU to operate over. See the docstring in
`src/models/model_factory.py` for the full mathematical justification.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Place the KOA-PD-NM dataset (Mendeley DOI 10.17632/44pfnysy89.1) videos under:

```
data/raw_videos/KOA/{KOA_EL,KOA_MD,KOA_SV}/*.mov
data/raw_videos/PD/{PD_EL,PD_MD,PD_SV}/*.mov
data/raw_videos/NM/*.mov
```

> The filename-based subject/severity parser in `src/utils/helpers.py`
> (`extract_subject_id`, `parse_class_and_severity`) is a best-effort
> heuristic. If your copy of the dataset uses different filenames, either
> adjust the regex there or (safer) hand-build
> `data/metadata/subjects.csv` mapping filename → subject_id → class →
> severity and point `preprocessing.py` at it instead.

## Running the pipeline

```bash
# 1) Generate MHIs from raw videos and create a subject-wise train/val/test split
python -m src.preprocessing.preprocessing

# 2) Train (choose a backbone: vgg16 | mobilenet | densenet)
python -m src.training.train --backbone mobilenet

# 3) Evaluate on the held-out test set
python -m src.evaluation.evaluate --backbone mobilenet

# ...or run all three stages back-to-back:
python -m src.main --stage all --backbone mobilenet
```

## Avoiding data leakage

Splitting is done **subject-wise, not video-wise** (`preprocessing.subject_wise_split`).
Each subject contributes two videos (walking in both directions); a naive
random split over videos can put one direction of a subject in train and
the other in test, letting the model partially learn to "recognize the
person" rather than the gait pathology. See the docstring in
`src/preprocessing/preprocessing.py` for details.

## Outputs

| Path | Contents |
|---|---|
| `models/checkpoints/` | best-epoch weights per training stage |
| `models/<backbone>_final_model.keras` | final trained model |
| `outputs/training_curves/` | accuracy/loss curves (both training stages) |
| `outputs/confusion_matrices/` | raw + normalized confusion matrices |
| `outputs/roc_curves/` | one-vs-rest ROC curves + AUC |
| `outputs/predictions/` | per-sample test-set predictions CSV |
| `data/metadata/` | subjects, labels, split assignment, dataset stats |

## Project structure

```
src/
├── preprocessing/   # video_loader, mhi_generator, preprocessing, augmentation
├── models/           # vgg16, mobilenet, densenet, model_factory
├── training/         # trainer, callbacks, train.py (entry point)
├── evaluation/       # metrics, confusion_matrix, roc_curve, plots, evaluate.py
├── utils/            # config, seed, logger, helpers
└── main.py           # orchestrates preprocess -> train -> evaluate
```
