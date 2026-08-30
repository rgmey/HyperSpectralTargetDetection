import time
import yaml
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # '3' hides INFO, WARNING, ERROR
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Reshape, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Data Loader
# -----------------------------
class HSIDataLoader:
    """Handles loading and preprocessing of hyperspectral image datasets."""

    DATASETS = {
        "IP": ("Indian_pines_corrected.mat", "indian_pines_corrected", "Indian_pines_gt.mat", "indian_pines_gt"),
        "SD": ("sandiego_reflectance.mat", "b", "sandiego_targetmap.mat", "sandiego_targetmap"),
        "SAA": ("SalinasA.mat", "salinasA", "SalinasA_gt.mat", "salinasA_gt"),
        "HP": ("HyMap.mat", "self_test_refl_sub", "HyMap_GT.mat", "self_test_targetmap_sub"),
    }

    @staticmethod
    def load_dataset(dataset_name: str, data_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load hyperspectral data and labels from .mat files."""
        if dataset_name not in HSIDataLoader.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        data_file, data_key, label_file, label_key = HSIDataLoader.DATASETS[dataset_name]
        data = sio.loadmat(f"{data_path}{data_file}")[data_key]
        labels = sio.loadmat(f"{data_path}{label_file}")[label_key]
        return data, labels

    @staticmethod
    def reduce_dimensionality(
        data_flat: np.ndarray,
        method: str,
        num_components: int,
        seed_value: int,
    ) -> tuple[np.ndarray, float]:
        """Reduce spectral dimensionality using PCA, t-SNE, or UMAP.

        All three methods are fit on the full flattened image (all pixels,
        train + test together) and applied via fit_transform, matching the
        original PCA behavior in this codebase. Note this differs from a
        strict train-only-fit protocol (see README for details); t-SNE in
        particular has no out-of-sample `.transform()`, so a train-only fit
        would require a different evaluation setup entirely if you want to
        avoid this.

        Returns the reduced array and the wall-clock time (seconds) taken
        by the reduction step, so reduction cost can be compared across
        methods the way the accompanying paper reports it.
        """
        method = method.lower()
        start_time = time.time()

        if method == "pca":
            reducer = PCA(n_components=num_components, whiten=True, random_state=seed_value)
            data_reduced = reducer.fit_transform(data_flat)

        elif method == "tsne":
            # t-SNE has no .transform() for new/unseen points, so it must be
            # fit_transform'd on the full array in one call. n_components > 3
            # requires method="exact" in scikit-learn (the default
            # barnes_hut backend only supports up to 3 output dimensions).
            tsne_method = "barnes_hut" if num_components <= 3 else "exact"
            reducer = TSNE(
                n_components=num_components,
                random_state=seed_value,
                method=tsne_method,
                init="pca",
            )
            data_reduced = reducer.fit_transform(data_flat)

        elif method == "umap":
            reducer = umap.UMAP(n_components=num_components, random_state=seed_value)
            data_reduced = reducer.fit_transform(data_flat)

        else:
            raise ValueError(f"Unknown reduction method: {method!r}. Expected 'pca', 'tsne', or 'umap'.")

        reduction_time = time.time() - start_time
        return data_reduced, reduction_time

    @staticmethod
    def preprocess_data(
        data: np.ndarray,
        labels: np.ndarray,
        target_class_num: int,
        num_components: int,
        window_size: int,
        test_ratio: float,
        seed_value: int,
        reduction_method: str = "pca",
    ) -> tuple:
        """Preprocess data: dimensionality reduction, patch extraction, and train-test split."""
        labels_binary = np.where(labels == target_class_num, 1, 0)
        labels_binary_1d = labels_binary.reshape(-1)

        num_target = int(labels_binary_1d.sum())
        num_background = int(labels_binary_1d.size - num_target)
        print(f"Class balance for target_class_num={target_class_num}: "
              f"{num_target} target pixels, {num_background} background pixels")
        if num_target == 0:
            raise ValueError(
                f"target_class_num={target_class_num} matches zero pixels in this dataset's "
                f"label map. Check that this class number actually exists for the selected "
                f"dataset (see the paper / README for the correct target class per dataset) "
                f"before running further."
            )

        data_flat = data.reshape(-1, data.shape[2])
        data_reduced, reduction_time = HSIDataLoader.reduce_dimensionality(
            data_flat, reduction_method, num_components, seed_value
        )
        print(f"[{reduction_method.upper()}] dimensionality reduction took {reduction_time:.2f}s")
        data_reduced_reshaped = data_reduced.reshape(data.shape[0], data.shape[1], num_components)
        data_patched = patch_data(data_reduced_reshaped, window_size)
        X_train, X_test, y_train, y_test = train_test_split(
            data_patched, labels_binary_1d, test_size=test_ratio, random_state=seed_value,
            stratify=labels_binary_1d,
        )
        return X_train, X_test, y_train, y_test, data_patched, reduction_time


# -----------------------------
# Utils
# -----------------------------
def pad_with_zeros(data: np.ndarray, margin: int) -> np.ndarray:
    """Pad 3D array with zeros on all sides."""
    new_shape = (data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2])
    padded = np.zeros(new_shape)
    padded[margin : data.shape[0] + margin, margin : data.shape[1] + margin, :] = data
    return padded


def patch_data(data: np.ndarray, window_size: int) -> np.ndarray:
    """Extract patches from 3D data array."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")

    margin = (window_size - 1) // 2
    data_padded = pad_with_zeros(data, margin)
    patched = np.zeros((data.shape[0] * data.shape[1], window_size, window_size, data.shape[2]))

    patch_index = 0
    for row in range(margin, data_padded.shape[0] - margin):
        for col in range(margin, data_padded.shape[1] - margin):
            patched[patch_index] = data_padded[
                row - margin : row + margin + 1, col - margin : col + margin + 1
            ]
            patch_index += 1
    return patched


# -----------------------------
# Model
# -----------------------------
def build_model(window_size: int, num_components: int, learning_rate: float, output_units: int = 1) -> Model:
    """Build and compile a 3D-2D CNN model for target detection."""
    input_layer = Input((window_size, window_size, num_components, 1))
    conv3d = Conv3D(filters=2, kernel_size=(3, 3, 1), activation="relu")(input_layer)
    conv3d_shape = conv3d.shape
    reshaped = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv3d)
    conv2d = Conv2D(filters=4, kernel_size=(3, 3), activation="relu")(reshaped)
    flatten = Flatten()(conv2d)
    dense1 = Dense(units=256, activation="relu")(flatten)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=128, activation="relu")(dense1)
    dense2 = Dropout(0.2)(dense2)
    output_layer = Dense(units=output_units, activation="sigmoid")(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Training
# -----------------------------
def train_model(config: dict, X_train, y_train, X_test, y_test, reduction_method: str = "pca"):
    """Train the model and return trained model, history, and training time."""
    model = build_model(config["window_size"], config["num_components"], config["learning_rate"])
    model.summary()

    start_time = time.time()
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"])],
    )
    execution_time = time.time() - start_time
    print(f"Model training time: {execution_time:.2f} seconds")

    os.makedirs('../models/', exist_ok=True)
    model.save("../models/TargetDetection_{}_{}_{}.keras".format(config["dataset"], reduction_method, config['target_class_num']))
    return model, history, execution_time


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, history, X_test, y_test, labels, data_patched, config, reduction_method: str = "pca"):
    """Evaluate model and plot metrics."""
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, np.round(y_pred, 0))
    recall = recall_score(y_test, np.round(y_pred, 0))
    f1score = f1_score(y_test, np.round(y_pred, 0))
    auc = roc_auc_score(y_test, y_pred)
    cls_report = classification_report(y_test, np.round(y_pred, 0), digits=4)
    print(f"\n--- Results: dataset={config['dataset']}  method={reduction_method}  "
          f"target_class={config['target_class_num']} ---")
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL: {recall:.4f}")
    print(f"F1 Score: {f1score:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"CLASSIFICATION REPORT:\n{cls_report}")
    os.makedirs('../plots/', exist_ok=True)
    df = pd.DataFrame(classification_report(y_test, np.round(y_pred, 0), digits=4, output_dict=True)).transpose()
    df.to_csv('../plots/df_cls_report_{}_{}_{}.csv'.format(config["dataset"], reduction_method, config['target_class_num']))

    plot_results(y_test, y_pred, history, labels, data_patched, config, model, reduction_method)

# -----------------------------
# Plot
# -----------------------------
def plot_results(y_test, y_pred, history, labels, data_patched, config, model, reduction_method: str = "pca"):
    # print(history.history.keys())
    os.makedirs('../plots/', exist_ok=True) 

    """Generate, save and display evaluation plots."""
    cm = confusion_matrix(y_test, np.round(y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # plt.show()
    plt.savefig('../plots/CM_{}_{}_{}.png'.format(config["dataset"], reduction_method, config['target_class_num']))
    plt.close()  # close to free memory

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.figure()  # create a fresh figure for each metric
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.legend()
    # plt.show()
    plt.savefig('../plots/AUC_{}_{}_{}.png'.format(config["dataset"], reduction_method, config['target_class_num']))
    plt.close()  # close to free memory

    patched_reshaped = data_patched.reshape(
        data_patched.shape[0], config["window_size"], config["window_size"], config["num_components"], 1
    )
    y_pred_final = model.predict(patched_reshaped)
    result_2d = np.round(y_pred_final).reshape(labels.shape[0], labels.shape[1])
    plt.figure()  # create a fresh figure for each metric
    plt.imshow(result_2d, cmap="gray")
    # plt.show()
    plt.savefig('../plots/PRED_{}_{}_{}.png'.format(config["dataset"], reduction_method, config['target_class_num']))
    plt.close()  # close to free memory

    for metric in ["accuracy", "loss"]:
        plt.figure()  # create a fresh figure for each metric
        # print(history.history[metric])
        plt.plot(history.history[metric], label=f"Training {metric.upper()}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric.upper()}")
        plt.legend()
        plt.title(metric.upper())
        # plt.show()
        plt.savefig('../plots/{}_{}_{}_{}.png'.format(metric.upper(), config["dataset"], reduction_method, config['target_class_num']))
        plt.close()  # close to free memory

    plt.figure()  # create a fresh figure for each metric
    plt.imshow(labels, cmap="gray")
    plt.title("Ground Truth")
    # plt.show()
    plt.savefig('../plots/GT_{}_{}.png'.format(config["dataset"], config['target_class_num']))
    plt.close()  # close to free memory


# -----------------------------
# Main
# -----------------------------
def main():
    config = load_config("config.yaml")
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    reduction_method = config.get("reduction_method", "pca")  # 'pca' (default), 'tsne', or 'umap'

    run_label = (f"dataset={config['dataset']}  method={reduction_method}  "
                 f"target_class={config['target_class_num']}")
    print("=" * 70)
    print(f"RUN START  |  {run_label}")
    print("=" * 70)

    # Load + preprocess
    data, labels = HSIDataLoader.load_dataset(config["dataset"], config["data_path"])
    X_train, X_test, y_train, y_test, data_patched, reduction_time = HSIDataLoader.preprocess_data(
        data, labels, config["target_class_num"], config["num_components"],
        config["window_size"], config["test_ratio"], config["seed_value"],
        reduction_method,
    )

    # Reshape for model input
    X_train = X_train.reshape(-1, config["window_size"], config["window_size"], config["num_components"], 1)
    X_test = X_test.reshape(-1, config["window_size"], config["window_size"], config["num_components"], 1)

    # Train
    model, history, training_time = train_model(config, X_train, y_train, X_test, y_test, reduction_method)
    print(f"Total time (reduction + training): {reduction_time + training_time:.2f}s")

    # Evaluate
    evaluate_model(model, history, X_test, y_test, labels, data_patched, config, reduction_method)

    print("=" * 70)
    print(f"RUN COMPLETE  |  {run_label}")
    print("=" * 70)