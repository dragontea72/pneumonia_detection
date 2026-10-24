"""
evaluate.py
-----------
Evaluate the trained models on the UNTOUCHED dataset/test folder and report
Accuracy, Precision, Recall/Sensitivity, F1-score, Specificity, ROC-AUC and
the Confusion Matrix.

Run (VS Code terminal, from the project folder):
    python evaluate.py
    python evaluate.py --threshold 0.5
    python evaluate.py --use-final       # evaluate the end-of-training models
                                         # instead of the best checkpoints

Outputs (in results/):
    confusion_matrix_<model>.png
    classification_report_<model>.txt
    predictions_<model>.csv              per-image probabilities
    roc_curves.png                       both models on one plot
    metrics_summary.csv                  side-by-side comparison table
"""

import argparse
import csv
from pathlib import Path

import utils  # import first: it silences TensorFlow start-up messages
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import matplotlib
matplotlib.use("Agg")                  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt


BEST_MODELS = {
    "Custom CNN": "best_custom_cnn.keras",
    "MobileNetV2": "best_mobilenetv2.keras",
}
FINAL_MODELS = {
    "Custom CNN": "custom_cnn.keras",
    "MobileNetV2": "mobilenetv2.keras",
}


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """PNEUMONIA (label 1) is the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    return metrics


def plot_confusion_matrix(cm, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=utils.CLASS_NAMES)
    ax.set_yticks([0, 1], labels=utils.CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / row_totals[i, 0] if row_totals[i, 0] else 0
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained models on the test set.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="probability above which an image is called PNEUMONIA")
    p.add_argument("--use-final", action="store_true",
                   help="evaluate end-of-training models instead of best checkpoints")
    p.add_argument("--batch-size", type=int, default=utils.BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    utils.setup()

    test_ds, file_paths = utils.load_test_dataset(args.batch_size)
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int).ravel()
    print(f"Test images: {len(y_true)}  "
          f"(NORMAL={int((y_true == 0).sum())}, PNEUMONIA={int((y_true == 1).sum())})")

    model_files = FINAL_MODELS if args.use_final else BEST_MODELS
    summary_rows = []
    roc_data = {}

    for name, filename in model_files.items():
        path = utils.MODELS_DIR / filename
        if not path.exists():
            print(f"\n[skip] {name}: {path.name} not found - train it first.")
            continue

        print(f"\n{'=' * 60}\n{name}  ({path.name})\n{'=' * 60}")
        model = tf.keras.models.load_model(path, compile=False)
        y_prob = model.predict(test_ds, verbose=1).ravel()
        y_pred = (y_prob >= args.threshold).astype(int)

        m = compute_metrics(y_true, y_pred, y_prob)
        report = classification_report(y_true, y_pred, labels=[0, 1],
                                       target_names=utils.CLASS_NAMES,
                                       digits=4, zero_division=0)
        print(report)
        print(f"Specificity: {m['specificity']:.4f}   ROC-AUC: {m['roc_auc']:.4f}")
        print(f"Confusion matrix  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  TP={m['TP']}")
        print(f"-> {m['FN']} pneumonia case(s) were missed (false negatives).")

        s = slug(name)
        with open(utils.RESULTS_DIR / f"classification_report_{s}.txt", "w") as f:
            f.write(f"{name} ({path.name}), threshold = {args.threshold}\n\n")
            f.write(report)
            f.write(f"\nSpecificity: {m['specificity']:.4f}\nROC-AUC: {m['roc_auc']:.4f}\n")

        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        plot_confusion_matrix(
            cm,
            f"{name}\nAcc {m['accuracy']:.3f} | Recall {m['recall_sensitivity']:.3f}"
            f" | F1 {m['f1_score']:.3f}",
            utils.RESULTS_DIR / f"confusion_matrix_{s}.png",
        )

        if file_paths:
            with open(utils.RESULTS_DIR / f"predictions_{s}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "true_label", "predicted_label",
                            "pneumonia_probability", "correct"])
                for fp, t, p, pr in zip(file_paths, y_true, y_pred, y_prob):
                    w.writerow([Path(fp).name, utils.CLASS_NAMES[t], utils.CLASS_NAMES[p],
                                f"{pr:.4f}", t == p])

        roc_data[name] = y_prob
        summary_rows.append({"model": name, "file": path.name, **m})

    if not summary_rows:
        print("\nNo trained models found in models/. "
              "Run train_cnn.py and/or transfer_learning.py first.")
        return

    # ROC curves of all evaluated models on one figure
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves - test set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(utils.RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Comparison table
    with open(utils.RESULTS_DIR / "metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    print(f"\n{'=' * 60}\nSUMMARY (test set, threshold = {args.threshold})\n{'=' * 60}")
    header = f"{'Model':<14}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Spec':>8}{'AUC':>8}"
    print(header)
    for r in summary_rows:
        print(f"{r['model']:<14}{r['accuracy']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall_sensitivity']:>8.4f}{r['f1_score']:>8.4f}"
              f"{r['specificity']:>8.4f}{r['roc_auc']:>8.4f}")
    print("\nAll figures, reports and metrics_summary.csv are in the results/ folder.")


if __name__ == "__main__":
    main()
