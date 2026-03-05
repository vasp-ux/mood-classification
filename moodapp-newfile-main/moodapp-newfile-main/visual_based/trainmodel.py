import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.utils import to_categorical

SEED = 42
IMG_SIZE = 48
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt",
]


def _default_dataset_candidates(base_dir):
    project_root = Path(base_dir).parent
    return [
        Path(base_dir) / "fer2013" / "train_balanced",
        Path(base_dir) / "fer2013" / "train",
        project_root / "mood-classification-main" / "visual" / "fer2013" / "train_balanced",
        project_root / "mood-classification-main" / "visual" / "fer2013" / "train",
    ]


def resolve_dataset_path(base_dir, dataset_arg):
    if dataset_arg:
        dataset_path = Path(dataset_arg).expanduser().resolve()
        if dataset_path.exists():
            return dataset_path
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    for candidate in _default_dataset_candidates(base_dir):
        if candidate.exists():
            return candidate.resolve()

    searched = [str(p) for p in _default_dataset_candidates(base_dir)]
    raise FileNotFoundError(
        "Could not find FER dataset directory. Tried:\n- " + "\n- ".join(searched)
    )


def load_dataset(dataset_path, max_per_class):
    rng = random.Random(SEED)
    x_data = []
    y_data = []
    class_counts = {}

    for class_index, emotion in enumerate(EMOTIONS):
        class_dir = dataset_path / emotion
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = [p for p in class_dir.iterdir() if p.is_file()]
        rng.shuffle(images)
        if max_per_class and max_per_class > 0:
            images = images[:max_per_class]

        kept = 0
        for image_path in images:
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            x_data.append(gray)
            y_data.append(class_index)
            kept += 1

        class_counts[emotion] = kept

    x_data = np.asarray(x_data, dtype="float32") / 255.0
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.asarray(y_data, dtype="int32")
    return x_data, y_data, class_counts


def build_model(num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.10),
            layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.20),
            layers.SeparableConv2D(64, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.SeparableConv2D(128, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.30),
            layers.SeparableConv2D(256, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def save_labels(base_dir):
    labels_path = Path(base_dir) / "emotion_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(EMOTIONS, f, indent=2)
    return labels_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train visual emotion model with balanced augmentation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="FER train directory containing emotion folders.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap for samples per class (0 = use all).",
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    dataset_path = resolve_dataset_path(base_dir, args.dataset)
    print(f"Dataset path: {dataset_path}")

    x_data, y_data, class_counts = load_dataset(dataset_path, args.max_per_class)
    print("Class counts:")
    for emotion in EMOTIONS:
        print(f"  {emotion:10s} {class_counts.get(emotion, 0)}")

    print(f"\nTotal samples: {len(x_data)}")
    print(f"Input shape: {x_data.shape}")

    y_categorical = to_categorical(y_data, num_classes=len(EMOTIONS))

    x_train, x_val, y_train, y_val, y_train_raw, y_val_raw = train_test_split(
        x_data,
        y_categorical,
        y_data,
        test_size=args.val_size,
        random_state=SEED,
        stratify=y_data,
    )

    classes = np.unique(y_train_raw)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_raw,
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
    print("\nClass weights:", class_weights)

    model = build_model(num_classes=len(EMOTIONS))
    model.summary()

    model_path = base_dir / "emotion_model.keras"
    log_path = base_dir / "training_history.csv"

    train_callbacks = [
        callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(str(log_path)),
    ]

    print("\nTraining started...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=train_callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")

    labels_path = save_labels(base_dir)
    print(f"Model saved at: {model_path}")
    print(f"Label map saved at: {labels_path}")
    print(f"Training log saved at: {log_path}")


if __name__ == "__main__":
    main()
