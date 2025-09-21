import time
import yaml
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # '3' hides INFO, WARNING, ERROR
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Reshape, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


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
        "SA": ("Salinas_corrected.mat", "salinas_corrected", "Salinas_gt.mat", "salinas_gt"),
        "PU": ("PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt"),
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
    def preprocess_data(
        data: np.ndarray,
        labels: np.ndarray,
        target_class_num: int,
        num_components: int,
        window_size: int,
        test_ratio: float,
        seed_value: int,
    ) -> tuple:
        """Preprocess data: PCA, patch extraction, and train-test split."""
        labels_binary = np.where(labels == target_class_num, 1, 0)
        labels_binary_1d = labels_binary.reshape(-1)
        data_flat = data.reshape(-1, data.shape[2])
        pca = PCA(n_components=num_components, whiten=True)
        data_pca = pca.fit_transform(data_flat)
        data_pca_reshaped = data_pca.reshape(data.shape[0], data.shape[1], num_components)
        data_patched = patch_data(data_pca_reshaped, window_size)
        X_train, X_test, y_train, y_test = train_test_split(
            data_patched, labels_binary_1d, test_size=test_ratio, random_state=seed_value
        )
        return X_train, X_test, y_train, y_test, data_patched


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
def train_model(config: dict, X_train, y_train, X_test, y_test):
    """Train the model and return trained model + history."""
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

    os.makedirs('models/', exist_ok=True) 
    model.save(config["model_name"].format(config["dataset"], config['target_class_num']))
    return model, history


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, history, X_test, y_test, labels, data_patched, config):
    """Evaluate model and plot metrics."""
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, np.round(y_pred, 0))
    recall = recall_score(y_test, np.round(y_pred, 0))
    f1score = f1_score(y_test, np.round(y_pred, 0))
    auc = roc_auc_score(y_test, y_pred)

    print(f"PRECISION: {precision:.3f}")
    print(f"RECALL: {recall:.3f}")
    print(f"F1 Score: {f1score:.3f}")
    print(f"AUC: {auc:.3f}")

    # plot_results(y_test, y_pred, history, labels, data_patched, config, model)

# -----------------------------
# Plot
# -----------------------------
def plot_results(y_test, y_pred, history, labels, data_patched, config, model):
    """Generate and display evaluation plots."""
    cm = confusion_matrix(y_test, np.round(y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.legend()
    plt.show()

    patched_reshaped = data_patched.reshape(
        data_patched.shape[0], config["window_size"], config["window_size"], config["num_components"], 1
    )
    y_pred_final = model.predict(patched_reshaped)
    result_2d = np.round(y_pred_final).reshape(labels.shape[0], labels.shape[1])
    plt.imshow(result_2d, cmap="gray")
    plt.show()

    for metric in ["accuracy", "loss"]:
        plt.plot(history.history[metric], label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.legend()
        plt.title(metric)
        plt.show()

    plt.imshow(labels, cmap="gray")
    plt.title("Ground Truth")
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    config = load_config("config.yaml")
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    # Load + preprocess
    data, labels = HSIDataLoader.load_dataset(config["dataset"], config["data_path"])
    X_train, X_test, y_train, y_test, data_patched = HSIDataLoader.preprocess_data(
        data, labels, config["target_class_num"], config["num_components"],
        config["window_size"], config["test_ratio"], config["seed_value"],
    )

    # Reshape for model input
    X_train = X_train.reshape(-1, config["window_size"], config["window_size"], config["num_components"], 1)
    X_test = X_test.reshape(-1, config["window_size"], config["window_size"], config["num_components"], 1)

    # Train
    model, history = train_model(config, X_train, y_train, X_test, y_test)

    # Evaluate
    evaluate_model(model, history, X_test, y_test, labels, data_patched, config)
