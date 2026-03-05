import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

IMG_SIZE = 48
DEFAULT_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt",
]


def resolve_dataset(base_dir, dataset_arg):
    if dataset_arg:
        dataset = Path(dataset_arg).expanduser().resolve()
        if dataset.exists():
            return dataset
        raise FileNotFoundError(f"Dataset path not found: {dataset}")

    candidates = [
        Path(base_dir) / "fer2013" / "train_balanced",
        Path(base_dir) / "fer2013" / "train",
        Path(base_dir).parent / "mood-classification-main" / "visual" / "fer2013" / "train_balanced",
        Path(base_dir).parent / "mood-classification-main" / "visual" / "fer2013" / "train",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find dataset directory.")


def load_labels(base_dir, model):
    labels_path = Path(base_dir) / "emotion_labels.json"
    expected = int(model.output_shape[-1])
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if (
                isinstance(labels, list)
                and len(labels) == expected
                and all(isinstance(item, str) for item in labels)
            ):
                return [item.strip().lower() for item in labels]
        except Exception:
            pass
    return DEFAULT_EMOTIONS[:expected]


def load_data(dataset_path, emotions, max_per_class):
    x_data = []
    y_data = []

    for idx, emotion in enumerate(emotions):
        class_dir = dataset_path / emotion
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        files = [p for p in class_dir.iterdir() if p.is_file()]
        if max_per_class and max_per_class > 0:
            files = files[:max_per_class]

        for file_path in files:
            gray = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            x_data.append(gray)
            y_data.append(idx)

    x_data = np.asarray(x_data, dtype="float32") / 255.0
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.asarray(y_data, dtype="int32")
    return x_data, y_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate visual emotion model.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--val-size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "emotion_model.keras"
    model = tf.keras.models.load_model(model_path, compile=False)

    emotions = load_labels(base_dir, model)
    dataset_path = resolve_dataset(base_dir, args.dataset)
    print(f"Dataset path: {dataset_path}")
    print(f"Labels: {emotions}")

    x_data, y_data = load_data(dataset_path, emotions, args.max_per_class)
    print(f"Loaded samples: {len(x_data)}")

    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=args.val_size,
        random_state=42,
        stratify=y_data,
    )

    probs = model.predict(x_val, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc = float(np.mean(y_pred == y_val))
    print(f"\nValidation accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=emotions, digits=4))


if __name__ == "__main__":
    main()
