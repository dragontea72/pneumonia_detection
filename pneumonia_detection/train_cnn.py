"""
train_cnn.py
------------
Model 1: a custom Convolutional Neural Network trained from scratch
to classify chest X-rays as NORMAL or PNEUMONIA.

Run (VS Code terminal, from the project folder):
    python train_cnn.py
    python train_cnn.py --epochs 3          # quick test run

Outputs:
    models/best_custom_cnn.keras            best model (lowest validation loss)
    models/custom_cnn.keras                 model at the end of training
    results/custom_cnn_training_curves.png
    results/custom_cnn_history.json
    results/custom_cnn_training_log.csv
"""

import argparse

import utils  # import first: it silences TensorFlow start-up messages
import tensorflow as tf

layers = tf.keras.layers


def build_custom_cnn(input_shape=(*utils.IMG_SIZE, 3)) -> tf.keras.Model:
    """
    Input (224x224x3, pixel values 0-255)
      -> light augmentation (training only)
      -> rescale to 0-1
      -> 5 x [Conv2D -> BatchNorm -> ReLU -> MaxPool]
      -> GlobalAveragePooling -> Dropout -> Dense(128) -> Dropout
      -> Dense(1, sigmoid) = probability of PNEUMONIA
    """
    inputs = layers.Input(shape=input_shape, name="image")
    x = utils.build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255, name="normalize")(x)

    for i, filters in enumerate((32, 64, 128, 256, 256), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)

    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def parse_args():
    p = argparse.ArgumentParser(description="Train the custom CNN.")
    p.add_argument("--epochs", type=int, default=30,
                   help="maximum epochs (early stopping may stop sooner)")
    p.add_argument("--batch-size", type=int, default=utils.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--patience", type=int, default=6,
                   help="early-stopping patience (epochs)")
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable class weighting for the imbalanced dataset")
    return p.parse_args()


def main():
    args = parse_args()
    utils.setup()

    # 1. Data: 80% train / 20% validation from dataset/train
    train_ds, val_ds = utils.load_train_val_datasets(args.batch_size)
    class_weight = None if args.no_class_weights else utils.compute_class_weights()

    # 2. Model
    model = build_custom_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=utils.training_metrics(),
    )
    model.summary()

    # 3. Train
    callbacks = utils.get_callbacks(
        checkpoint_path=utils.MODELS_DIR / "best_custom_cnn.keras",
        csv_log_path=utils.RESULTS_DIR / "custom_cnn_training_log.csv",
        patience=args.patience,
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # 4. Save
    model.save(utils.MODELS_DIR / "custom_cnn.keras")
    utils.save_history(history.history, utils.RESULTS_DIR / "custom_cnn_history.json")
    utils.plot_history(history.history, "Custom CNN - training curves",
                       utils.RESULTS_DIR / "custom_cnn_training_curves.png")

    print("\nDone. Saved models/best_custom_cnn.keras and models/custom_cnn.keras")
    print("Next: python transfer_learning.py")


if __name__ == "__main__":
    main()
