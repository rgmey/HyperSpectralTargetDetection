# Hyperspectral Target Detection

A lightweight hybrid 3D–2D convolutional neural network for binary target detection in hyperspectral imagery, evaluated across three unsupervised dimensionality reduction methods (PCA, t-SNE, UMAP) and four benchmark datasets.

This repository implements the method described in *"Dimensionality Reduction based Convolutional Network for Hyperspectral Target Detection"* (Ghorbanmeyabadi, Imani, and Ghassemian).

## Overview

Hyperspectral target detection is challenging due to the high dimensionality of spectral data, limited labeled samples, and subtle spectral differences between target and background pixels. This project addresses that with:

1. **Dimensionality reduction** — spectral bands are reduced to a small number of components (default: 3) using one of three unsupervised methods: PCA, t-SNE, or UMAP, selectable via config.
2. **Spatial–spectral patch extraction** — a sliding window is applied around each pixel to capture local spatial context alongside the reduced spectral information.
3. **Lightweight hybrid 3D–2D CNN classification** — a 3D convolution layer extracts joint spectral–spatial features, followed by a 2D convolution for spatial refinement and dense layers for binary classification (target vs. background).

## Architecture

```
Hyperspectral Image
        │
        ▼
Dimensionality Reduction (PCA / t-SNE / UMAP → N components)
        │
        ▼
Sliding-window Patch Extraction (W × W × N)
        │
        ▼
3D Conv (spectral-spatial features) → 2D Conv (spatial refinement)
        │
        ▼
Flatten → Dense(256) → Dropout → Dense(128) → Dropout → Dense(1, sigmoid)
        │
        ▼
Target / Background
```

## Datasets

Configurable via `config.yaml`. Dataset specs and target class definitions (from the accompanying paper):

| Key   | Dataset      | Spatial Size | Spectral Bands | Classes                            | Target `target_class_num`                                 |
| ----- | ------------ | ------------ | -------------- | ---------------------------------- | --------------------------------------------------------- |
| `SD`  | San Diego    | 80 × 80      | 189            | 2 (target vs. background)          | `1` *(binary map; confirm via class-balance check below)* |
| `IP`  | Indian Pines | 145 × 145    | 200            | 16 land-cover classes + background | `16`                                                      |
| `HP`  | HyMap        | 150 × 150    | 126            | 2 (target vs. background)          | `1` *(binary map; confirm via class-balance check below)* |
| `SAA` | Salinas-A    | 86 × 83      | 224            | 7                                  | `12`                                                      |

Dataset `.mat` files are expected in `data_path` (see `config.yaml`); they are not included in this repository due to size and licensing.

**Before running a dataset/class combination for the first time**, check the console output on the first run: `preprocess_data` prints the target vs. background pixel count for the chosen `target_class_num`, and raises an error immediately if that class doesn't exist in the label map (rather than silently training on an empty or degenerate split). If you see a `ValueError` about zero target pixels, the class number for that dataset needs adjusting, the values above for `SD` and `HP` are inferred from the paper's "2 classes" description rather than stated explicitly, so double-check them against your own label files.

## Getting Started

```bash
git clone https://github.com/rgmey/HyperSpectralTargetDetection.git
cd HyperSpectralTargetDetection
pip install -r requirements.txt
cd src
python run.py
```

## Configuration

All experiment settings are controlled through `src/config.yaml`:

```yaml
dataset: SAA            # IP, SD, SAA, or HP
target_class_num: 10    # class treated as the target (all others = background)
reduction_method: pca   # pca, tsne, or umap
num_components: 3       # number of components after reduction
window_size: 5          # spatial patch size (must be odd)
test_ratio: 0.1
seed_value: 7
batch_size: 32
epochs: 10
learning_rate: 0.01
patience: 3             # early stopping patience
data_path: ../HSI_Datasets/
```

To reproduce a comparison across all three reduction methods on a given dataset, run `python run.py` three times, changing only `reduction_method` between runs. Outputs (model checkpoints, plots, and classification reports) are automatically tagged with the dataset, method, and target class, so results from different runs don't overwrite each other.

## Outputs

Each run produces, under `../plots/` and `../models/`:

- Trained model checkpoint (`.keras`)
- Confusion matrix
- ROC curve (with AUC)
- Predicted target map vs. ground truth
- Training/validation accuracy and loss curves
- Classification report (CSV: precision, recall, F1, accuracy)

Console output also reports dimensionality reduction time, training time, and total time per run, for comparing computational cost across methods.

## Methodology Note

Dimensionality reduction (PCA, t-SNE, and UMAP) is fit on the full image, all pixels, before the train/test split, applied uniformly across all three methods. This is a deliberate consistency choice: t-SNE has no out-of-sample `.transform()` for unseen points, so a strict train-only fit isn't directly possible for it without a different evaluation design. Since the reduction step is unsupervised, this does not leak label information, but it does mean the test set's spectral distribution contributes to the learned embedding. See the Limitations section of the paper for further discussion.

## Tech Stack

Python · TensorFlow / Keras · scikit-learn (PCA, t-SNE) · umap-learn · NumPy · SciPy · Matplotlib · Pandas

## Limitations / Future Work

- Dimensionality reduction is fit on the full image rather than training pixels only (see Methodology Note above)
- Reduction parameters (number of components, window size) are fixed per run rather than tuned per dataset/method
- Binary classification only (target vs. background); no multiclass extension
- No standardized comparison against external state-of-the-art detectors (see paper for discussion)
