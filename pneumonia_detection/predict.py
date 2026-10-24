"""
predict.py
----------
Predict NORMAL vs PNEUMONIA for a single chest X-ray (or every image in a folder)
using a saved model.

Run (VS Code terminal, from the project folder):
    python predict.py                                   # random image from dataset/test
    python predict.py path/to/xray.jpeg
    python predict.py path/to/folder_of_xrays
    python predict.py path/to/xray.jpeg --model models/best_custom_cnn.keras
    python predict.py path/to/xray.jpeg --show          # also display the image

Educational project only - not a medical diagnostic tool.
"""

import argparse
import random
from pathlib import Path

import utils  # import first: it silences TensorFlow start-up messages
import numpy as np
import tensorflow as tf

DEFAULT_MODEL = utils.MODELS_DIR / "best_mobilenetv2.keras"


def load_image(path: Path) -> np.ndarray:
    """
    Load an image with EXACTLY the same steps image_dataset_from_directory used
    during training (decode -> 3 channels -> TensorFlow bilinear resize), so the
    prediction here matches the probability reported by evaluate.py.
    """
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, utils.IMG_SIZE, method="bilinear")
    return img.numpy()                                # shape (224, 224, 3), values 0-255


def predict_one(model, path: Path, threshold: float):
    """Return (label, confidence, pneumonia_probability)."""
    x = np.expand_dims(load_image(path), axis=0)      # add batch dimension
    prob = float(model.predict(x, verbose=0)[0][0])   # sigmoid output = P(PNEUMONIA)
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1.0 - prob
    return label, confidence, prob


def random_test_image() -> Path:
    candidates = [p for c in utils.CLASS_NAMES for p in (utils.TEST_DIR / c).glob("*")
                  if p.suffix.lower() in utils.IMAGE_EXTENSIONS]
    if not candidates:
        raise SystemExit("No image given and no images found in dataset/test. "
                         "Usage: python predict.py path/to/xray.jpeg")
    return random.choice(candidates)


def show_image(path: Path, label: str, confidence: float) -> None:
    import matplotlib.pyplot as plt
    img = tf.keras.utils.load_img(path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(f"{label}  ({confidence:.1%} confidence)",
              color="red" if label == "PNEUMONIA" else "green", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Predict NORMAL vs PNEUMONIA for chest X-rays.")
    p.add_argument("image", nargs="?", default="",
                   help="image file or folder (default: a random dataset/test image)")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="path to a saved .keras model")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--show", action="store_true", help="display the image with the prediction")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute() and not model_path.exists():
        model_path = utils.PROJECT_DIR / model_path
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\n"
                         "Train it first (python transfer_learning.py) "
                         "or pass --model models/best_custom_cnn.keras")

    # Work out which image(s) to predict
    if args.image:
        target = Path(args.image).expanduser()
        if not target.exists():
            raise SystemExit(f"Image or folder not found: {target}")
        images = (sorted(p for p in target.iterdir()
                         if p.suffix.lower() in utils.IMAGE_EXTENSIONS)
                  if target.is_dir() else [target])
    else:
        images = [random_test_image()]
        print(f"No image given - using a random test image "
              f"(true class: {images[0].parent.name})")

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model: {model_path.name}   threshold: {args.threshold}\n")

    for path in images:
        label, confidence, prob = predict_one(model, path, args.threshold)
        print(f"{path.name:<40} -> {label:<9}  confidence {confidence:6.1%}  "
              f"(P(pneumonia) = {prob:.4f})")

    if args.show and len(images) == 1:
        show_image(images[0], label, confidence)

    print("\nNote: educational project only - not a medical diagnostic tool.")


if __name__ == "__main__":
    main()
