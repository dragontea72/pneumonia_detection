"""
transfer_learning.py
--------------------
Model 2: MobileNetV2 pretrained on ImageNet, adapted to NORMAL vs PNEUMONIA.

Stage 1 - feature extraction:
    the pretrained backbone is frozen, only the new classification head trains.
Stage 2 - fine-tuning:
    the top layers of the backbone are unfrozen and trained with a much
    smaller learning rate.

Run (VS Code terminal, from the project folder):
    python transfer_learning.py
    python transfer_learning.py --epochs-frozen 2 --epochs-finetune 2   # quick test

The first run downloads the ImageNet weights (~9 MB), so you need internet.

Outputs:
    models/best_mobilenetv2.keras           best model across both stages
    models/mobilenetv2.keras                model at the end of training
    results/mobilenetv2_training_curves.png
    results/mobilenetv2_history.json
    results/mobilenetv2_training_log.csv
"""

import argparse

import utils  # import first: it silences TensorFlow start-up messages
import tensorflow as tf

layers = tf.keras.layers


def build_mobilenetv2(input_shape=(*utils.IMG_SIZE, 3)):
    """Return (full_model, backbone) so the backbone can be unfrozen later."""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,          # remove the 1000-class ImageNet head
        weights="imagenet",
    )
    backbone.trainable = False      # Stage 1: freeze the pretrained layers

    inputs = layers.Input(shape=input_shape, name="image")
    x = utils.build_augmentation()(inputs)
    # MobileNetV2 expects pixels in [-1, 1]. This Rescaling layer does exactly
    # what mobilenet_v2.preprocess_input does (x / 127.5 - 1), but it saves and
    # reloads cleanly inside the .keras file.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    # training=False keeps BatchNorm layers in inference mode, even when we
    # unfreeze the backbone later (recommended for fine-tuning).
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_pneumonia")
    return model, backbone


def unfreeze_top_layers(backbone: tf.keras.Model, n_layers: int) -> None:
    """Unfreeze the last n_layers of the backbone; keep BatchNorm frozen."""
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - n_layers)
    for i, layer in enumerate(backbone.layers):
        if i < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    n_trainable = sum(layer.trainable for layer in backbone.layers)
    print(f"\nFine-tuning: {n_trainable} of {len(backbone.layers)} backbone "
          f"layers are now trainable (BatchNorm layers stay frozen).")


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=utils.training_metrics(),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train MobileNetV2 with transfer learning.")
    p.add_argument("--epochs-frozen", type=int, default=10,
                   help="Stage 1 epochs (backbone frozen)")
    p.add_argument("--epochs-finetune", type=int, default=10,
                   help="Stage 2 epochs (top layers unfrozen)")
    p.add_argument("--unfreeze-layers", type=int, default=40,
                   help="how many of the backbone's top layers to fine-tune")
    p.add_argument("--lr", type=float, default=1e-3, help="Stage 1 learning rate")
    p.add_argument("--finetune-lr", type=float, default=1e-5, help="Stage 2 learning rate")
    p.add_argument("--batch-size", type=int, default=utils.BATCH_SIZE)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--skip-finetune", action="store_true", help="only run Stage 1")
    p.add_argument("--no-class-weights", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    utils.setup()

    train_ds, val_ds = utils.load_train_val_datasets(args.batch_size)
    class_weight = None if args.no_class_weights else utils.compute_class_weights()

    best_path = utils.MODELS_DIR / "best_mobilenetv2.keras"
    log_path = utils.RESULTS_DIR / "mobilenetv2_training_log.csv"

    # ---------------- Stage 1: train the new head only ----------------
    model, backbone = build_mobilenetv2()
    compile_model(model, args.lr)
    model.summary()

    print("\n=== Stage 1: feature extraction (backbone frozen) ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        class_weight=class_weight,
        callbacks=utils.get_callbacks(best_path, log_path, patience=args.patience),
    )
    history = {k: list(v) for k, v in history1.history.items()}
    stage1_epochs = len(history1.epoch)
    fine_tune_start = None

    # ---------------- Stage 2: fine-tune the top layers ----------------
    if not args.skip_finetune and args.epochs_finetune > 0:
        print("\n=== Stage 2: fine-tuning ===")
        unfreeze_top_layers(backbone, args.unfreeze_layers)
        compile_model(model, args.finetune_lr)   # must recompile after (un)freezing

        # Only overwrite the checkpoint if Stage 2 beats Stage 1's best val_loss.
        best_stage1 = min(history1.history["val_loss"])
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage1_epochs + args.epochs_finetune,
            initial_epoch=stage1_epochs,
            class_weight=class_weight,
            callbacks=utils.get_callbacks(
                best_path, log_path, patience=args.patience,
                initial_value_threshold=best_stage1, append_log=True),
        )
        for k, v in history2.history.items():
            history.setdefault(k, []).extend(v)
        fine_tune_start = stage1_epochs

    # ---------------- Save ----------------
    history.pop("learning_rate", None)
    history.pop("lr", None)
    model.save(utils.MODELS_DIR / "mobilenetv2.keras")
    utils.save_history(history, utils.RESULTS_DIR / "mobilenetv2_history.json")
    utils.plot_history(history, "MobileNetV2 transfer learning - training curves",
                       utils.RESULTS_DIR / "mobilenetv2_training_curves.png",
                       fine_tune_start=fine_tune_start)

    print("\nDone. Saved models/best_mobilenetv2.keras and models/mobilenetv2.keras")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
