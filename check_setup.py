"""
check_setup.py
--------------
Run this FIRST. It checks that Python, the packages and the dataset folders
are all set up correctly before you spend time training.

    python check_setup.py
"""

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TensorFlow import

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
# Kaggle archives commonly add a nested chest_xray/ directory.
if not ((DATASET_DIR / "train").is_dir() or (DATASET_DIR / "test").is_dir()):
    nested_dataset_dir = DATASET_DIR / "chest_xray"
    if nested_dataset_dir.is_dir():
        DATASET_DIR = nested_dataset_dir
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".gif"}

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  [X] {msg}")


print("=" * 60)
print("1. Python and packages")
print("=" * 60)
print(f"  Python {sys.version.split()[0]}  ({sys.executable})")
if sys.version_info < (3, 9):
    fail("Python 3.9 or newer is required (3.11 or 3.12 recommended).")

for module, pip_name in [("tensorflow", "tensorflow"), ("numpy", "numpy"),
                         ("matplotlib", "matplotlib"), ("sklearn", "scikit-learn"),
                         ("PIL", "pillow")]:
    try:
        mod = importlib.import_module(module)
        print(f"  [OK] {pip_name:<13} {getattr(mod, '__version__', '')}")
    except ImportError:
        fail(f"{pip_name} is not installed ->  pip install -r requirements.txt")
        print("      (If you already installed it, VS Code may be using a different "
              "Python interpreter: Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv)")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  Device: {'GPU x' + str(len(gpus)) if gpus else 'CPU only (training will be slower, but works)'}")
except ImportError:
    pass

print()
print("=" * 60)
print("2. Dataset folders")
print("=" * 60)
if not DATASET_DIR.is_dir():
    fail(f"'dataset' folder not found at {DATASET_DIR}")
else:
    for split in ["train", "test"]:
        split_dir = DATASET_DIR / split
        if not split_dir.is_dir():
            fail(f"dataset/{split} is missing")
            continue
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                fail(f"dataset/{split}/{cls} is missing (folder names are case-sensitive)")
                continue
            files = [f for f in cls_dir.iterdir() if f.is_file()]
            images = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
            others = [f for f in files if f not in images and not f.name.startswith(".")]
            status = "[OK]" if images else "[X] "
            print(f"  {status} dataset/{split}/{cls:<10} {len(images):>5} images")
            if not images:
                fail(f"no images found in dataset/{split}/{cls}")
            if others:
                print(f"       note: {len(others)} non-image file(s) will be ignored")

    # Common mistake: the Kaggle zip contains a nested chest_xray/ folder
    nested = [p for p in DATASET_DIR.rglob("chest_xray") if p.is_dir()]
    if nested:
        print(f"  note: found a nested folder {nested[0].relative_to(PROJECT_DIR)} - "
              "move its train/ and test/ folders directly into dataset/")
    extra = [p.name for p in DATASET_DIR.iterdir()
             if p.is_dir() and p.name not in ("train", "test") and not p.name.startswith(".")]
    if extra:
        print(f"  note: extra folder(s) {extra} in dataset/ will be ignored "
              "(validation is split from train/ automatically)")

    # Try opening one image per class
    try:
        from PIL import Image
        for cls in CLASS_NAMES:
            sample = next((f for f in (DATASET_DIR / "train" / cls).glob("*")
                           if f.suffix.lower() in IMAGE_EXTENSIONS), None)
            if sample:
                with Image.open(sample) as im:
                    print(f"  sample {cls:<10} {sample.name}: {im.size[0]}x{im.size[1]} {im.mode}")
    except Exception as e:  # noqa: BLE001
        fail(f"could not open a sample image: {e}")

print()
print("=" * 60)
if ok:
    print("All checks passed. Next:  python train_cnn.py")
else:
    print("Fix the [X] items above, then run  python check_setup.py  again.")
print("=" * 60)
