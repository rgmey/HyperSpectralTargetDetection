import time
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Reshape, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# HSIDataLoader class (unchanged)
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
    def preprocess_data(data: np.ndarray, labels: np.ndarray, class_num: int, num_components: int,
                       window_size: int, test_ratio: float, seed_value: int) -> tuple:
        """Preprocess data: PCA, patch extraction, and train-test split."""
        labels_binary = np.where(labels == class_num, 1, 0)
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

def pad_with_zeros(data: np.ndarray, margin: int) -> np.ndarray:
    """Pad 3D array with zeros on all sides."""
    new_shape = (data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2])
    padded = np.zeros(new_shape)
    padded[margin:data.shape[0] + margin, margin:data.shape[1] + margin, :] = data
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
                row - margin:row + margin + 1,
                col - margin:col + margin + 1
            ]
            patch_index += 1
    return patched

def build_model(window_size: int, num_components: int, output_units: int = 1) -> Model:
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
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        metrics=["accuracy"]
    )
    return model

def plot_results(y_test: np.ndarray, y_pred: np.ndarray, history: dict, labels: np.ndarray,
                 data_patched: np.ndarray, dataset_name: str, window_size: int, num_components: int) -> None:
    """Generate and display evaluation plots: confusion matrix, ROC curve, prediction map, accuracy/loss curves."""
    cm = confusion_matrix(y_test, np.round(y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(5, 5))
    disp.plot()
    plt.grid(False)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    patched_reshaped = data_patched.reshape(data_patched.shape[0], window_size, window_size, num_components, 1)
    y_pred_final = model.predict(patched_reshaped)
    result_2d = np.round(y_pred_final).reshape(labels.shape[0], labels.shape[1])
    plt.figure(figsize=(4, 4))
    plt.imshow(result_2d, cmap="gray")
    plt.grid(False)
    plt.show()

    for metric in ["accuracy", "loss"]:
        plt.figure(figsize=(3, 3))
        plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend(loc="lower right" if metric == "accuracy" else "upper right")
        plt.title(dataset_name)
        plt.show()

def main():
    """Main function to orchestrate HSI target detection pipeline."""
    # Load configuration
    global CONFIG
    CONFIG = load_config("config.yml")
    
    np.random.seed(CONFIG["seed_value"])
    tf.random.set_seed(CONFIG["seed_value"])

    # Load and preprocess data
    data, labels = HSIDataLoader.load_dataset(CONFIG["dataset"], CONFIG["data_path"])
    X_train, X_test, y_train, y_test, data_patched = HSIDataLoader.preprocess_data(
        data, labels, CONFIG["class_num"], CONFIG["num_components"],
        CONFIG["window_size"], CONFIG["test_ratio"], CONFIG["seed_value"]
    )

    # Reshape for model input
    X_train = X_train.reshape(-1, CONFIG["window_size"], CONFIG["window_size"], CONFIG["num_components"], 1)
    X_test = X_test.reshape(-1, CONFIG["window_size"], CONFIG["window_size"], CONFIG["num_components"], 1)

    # Build and train model
    global model
    model = build_model(CONFIG["window_size"], CONFIG["num_components"])
    model.summary()
    
    start_time = time.time()
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CONFIG["patience"])]
    )
    execution_time = time.time() - start_time

    # Save model
    model.save(CONFIG["model_name"].format(CONFIG["dataset"]))
    print(f"Model training time: {execution_time:.2f} seconds")

    # Evaluate and plot results
    y_pred = model.predict(X_test)
    plot_results(y_test, y_pred, history, labels, data_patched, CONFIG["dataset"],
                 CONFIG["window_size"], CONFIG["num_components"])

    # Display ground truth
    plt.figure(figsize=(4, 4))
    plt.imshow(labels, cmap="gray")
    plt.grid(False)
    plt.title("Ground Truth")
    plt.show()

if __name__ == "__main__":
    main()