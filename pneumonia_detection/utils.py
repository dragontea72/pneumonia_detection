"""
utils.py
--------
Shared settings and helper functions used by every script in this project.

Keeping these in one place means the image size, class order, dataset split
and preprocessing are guaranteed to be identical in training, evaluation and
prediction.
"""

import os

# Hide TensorFlow's long INFO / WARNING start-up messages.
# This must be set BEFORE TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Paths
# All paths are built relative to THIS file, so the scripts work no matter
# which folder VS Code (or the terminal) is currently in.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20          # 20% of dataset/train is used for validation

# Label 0 = NORMAL, label 1 = PNEUMONIA (the sigmoid output is P(PNEUMONIA)).
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".gif"}

AUTOTUNE = tf.data.AUTOTUNE

# Harmless Keras 3 warning: our tf.data pipeline already shuffles every epoch.
warnings.filterwarnings("ignore", message=".*`shuffle=True` was passed.*")


# ---------------------------------------------------------------------------
# Environment set-up
# ---------------------------------------------------------------------------
def setup(seed: int = SEED) -> None:
    """Fix random seeds, configure the GPU (if any) and create output folders."""
    tf.keras.utils.set_random_seed(seed)

    # Let TensorFlow allocate GPU memory gradually instead of grabbing all of it.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | "
          f"{'GPU: ' + str(len(gpus)) if gpus else 'running on CPU'}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def count_images(folder: Path) -> dict:
    """Return {class_name: number_of_images} for a train/ or test/ folder."""
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = folder / class_name
        if class_dir.is_dir():
            counts[class_name] = sum(
                1 for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            counts[class_name] = 0
    return counts


def _check_folder(folder: Path) -> None:
    """Stop early with a clear message if the dataset folder is missing/wrong."""
    missing = [c for c in CLASS_NAMES if not (folder / c).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"\nExpected sub-folders {CLASS_NAMES} inside:\n  {folder}\n"
            f"Missing: {missing}\n"
            "Run  python check_setup.py  to diagnose the dataset layout."
        )


def load_train_val_datasets(batch_size: int = BATCH_SIZE):
    """
    Load dataset/train and split it 80% training / 20% validation.
    The same seed is used in both calls so the two subsets never overlap.
    """
    _check_folder(TRAIN_DIR)
    common = dict(
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",            # X-rays are grayscale; repeat to 3 channels
        interpolation="bilinear",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR), subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR), subset="validation", **common)

    # Prefetch overlaps data loading with training for speed.
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def load_test_dataset(batch_size: int = BATCH_SIZE):
    """Load the untouched dataset/test folder (never shuffled)."""
    _check_folder(TEST_DIR)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        interpolation="bilinear",
        shuffle=False,
    )
    file_paths = list(getattr(test_ds, "file_paths", []))
    return test_ds.prefetch(AUTOTUNE), file_paths


def compute_class_weights() -> dict:
    """
    The chest X-ray dataset has far more PNEUMONIA than NORMAL images.
    Class weights make mistakes on the smaller class cost more during training.
    Weights are computed from the folder counts; the 80/20 split is random,
    so the class ratio in the training subset is essentially the same.
    """
    counts = count_images(TRAIN_DIR)
    total = sum(counts.values())
    weights = {
        i: total / (len(CLASS_NAMES) * max(counts[name], 1))
        for i, name in enumerate(CLASS_NAMES)
    }
    print("Class counts :", counts)
    print("Class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in weights.items()})
    return weights


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def build_augmentation() -> tf.keras.Sequential:
    """
    Light, medically sensible augmentation (only active during training).
    No vertical flips and no large rotations - those would create unrealistic
    chest X-rays.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),            # about +/- 10 degrees
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )


def training_metrics() -> list:
    """Fresh metric objects (a new list is needed every time we compile)."""
    return [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def get_callbacks(checkpoint_path: Path, csv_log_path: Path, patience: int = 5,
                  initial_value_threshold=None, append_log: bool = False) -> list:
    """Checkpoint the best model, stop early, lower the LR on plateaus, log to CSV."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=initial_value_threshold,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=append_log),
    ]


# ---------------------------------------------------------------------------
# Saving / plotting helpers
# ---------------------------------------------------------------------------
def save_history(history: dict, path: Path) -> None:
    """Save a Keras history dict as JSON (useful for the README later)."""
    clean = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def plot_history(history: dict, title: str, out_path: Path,
                 fine_tune_start: int = None) -> None:
    """Save loss / accuracy / precision / recall training curves as a PNG."""
    import matplotlib
    matplotlib.use("Agg")                      # save to file, no pop-up window
    import matplotlib.pyplot as plt

    panels = [("loss", "Loss"), ("accuracy", "Accuracy"),
              ("precision", "Precision"), ("recall", "Recall / Sensitivity")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, label) in zip(axes.ravel(), panels):
        if key not in history:
            ax.axis("off")
            continue
        epochs = np.arange(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], label="validation")
        if fine_tune_start is not None:
            ax.axvline(fine_tune_start + 0.5, color="gray", linestyle="--",
                       label="fine-tuning starts")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves -> {out_path.relative_to(PROJECT_DIR)}")
