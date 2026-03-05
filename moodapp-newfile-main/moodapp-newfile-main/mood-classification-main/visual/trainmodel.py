import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, losses, metrics, models, optimizers

SEED = 42
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
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_fer_candidates(base_dir: Path) -> list[Path]:
    return [
        base_dir / "fer2013" / "train_balanced",
        base_dir / "fer2013" / "train",
        base_dir / "fer2013",
    ]


def is_emotion_folder_root(path: Path) -> bool:
    return all((path / emotion).exists() for emotion in EMOTIONS)


def resolve_split_root(path: Path) -> Path:
    split_candidates = ["train_balanced", "train", "validation", "test"]
    for split in split_candidates:
        candidate = path / split
        if candidate.exists() and is_emotion_folder_root(candidate):
            return candidate
    return path


def resolve_fer_root(base_dir: Path, fer_root_arg: str) -> Path:
    if fer_root_arg:
        root = Path(fer_root_arg).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"FER root does not exist: {root}")
        normalized = resolve_split_root(root)
        if is_emotion_folder_root(normalized):
            return normalized
        raise FileNotFoundError(
            f"FER root has no emotion folders. Checked: {normalized}"
        )

    for candidate in default_fer_candidates(base_dir):
        if not candidate.exists():
            continue
        normalized = resolve_split_root(candidate.resolve())
        if is_emotion_folder_root(normalized):
            return normalized

    searched = "\n".join(f"- {path}" for path in default_fer_candidates(base_dir))
    raise FileNotFoundError(f"Could not find FER training folder. Tried:\n{searched}")


def collect_class_files(class_dir: Path, max_items: int) -> list[Path]:
    files = [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    files.sort()
    random.shuffle(files)

    if max_items > 0:
        files = files[:max_items]

    return files


def collect_dataset(
    fer_root: Path,
    extra_roots: list[str],
    max_fer_per_class: int,
    max_extra_per_class: int,
) -> tuple[list[str], np.ndarray, dict[str, dict[str, int]]]:
    image_paths: list[str] = []
    labels: list[int] = []
    counts: dict[str, dict[str, int]] = {
        emotion: {"fer": 0, "extra": 0} for emotion in EMOTIONS
    }

    for label, emotion in enumerate(EMOTIONS):
        fer_class_dir = fer_root / emotion
        if not fer_class_dir.exists():
            raise FileNotFoundError(f"Missing FER class folder: {fer_class_dir}")

        fer_files = collect_class_files(fer_class_dir, max_fer_per_class)
        for file_path in fer_files:
            image_paths.append(str(file_path))
            labels.append(label)
        counts[emotion]["fer"] = len(fer_files)

    for extra_root_raw in extra_roots:
        extra_root = Path(extra_root_raw).expanduser().resolve()
        if not extra_root.exists():
            print(f"[warn] Skipping missing extra root: {extra_root}")
            continue

        for label, emotion in enumerate(EMOTIONS):
            extra_class_dir = extra_root / emotion
            if not extra_class_dir.exists():
                continue

            extra_files = collect_class_files(extra_class_dir, max_extra_per_class)
            for file_path in extra_files:
                image_paths.append(str(file_path))
                labels.append(label)
            counts[emotion]["extra"] += len(extra_files)

    if not image_paths:
        raise RuntimeError("No training images were found.")

    labels_np = np.asarray(labels, dtype=np.int32)
    return image_paths, labels_np, counts


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, img_size: int, num_classes: int):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=num_classes)


def make_dataset(
    paths: list[str],
    labels: np.ndarray,
    img_size: int,
    num_classes: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda p, y: decode_and_resize(p, y, img_size, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(img_size: int, num_classes: int):
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(img_size, img_size, 1), name="gray_input")
    x = augmentation(inputs)
    x = layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t), name="to_rgb")(x)

    pretrained = True
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
    except Exception as exc:
        print(f"[warn] Could not load ImageNet weights ({exc}). Using random init backbone.")
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights=None,
        )
        pretrained = False

    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="emotion_mobilenetv2")
    return model, backbone, pretrained


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )


def unfreeze_top_layers(backbone: tf.keras.Model, layers_to_unfreeze: int) -> None:
    if layers_to_unfreeze <= 0:
        return

    backbone.trainable = True

    total_layers = len(backbone.layers)
    freeze_until = max(0, total_layers - layers_to_unfreeze)
    for idx, layer in enumerate(backbone.layers):
        if idx < freeze_until:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a more accurate visual emotion model using FER + extra labeled images.",
    )
    parser.add_argument(
        "--fer-root",
        type=str,
        default="",
        help="FER training root containing emotion folders.",
    )
    parser.add_argument(
        "--extra-data",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional extra dataset roots. Each root should contain subfolders: "
            + ", ".join(EMOTIONS)
        ),
    )
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-epochs", type=int, default=18)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--unfreeze-layers", type=int, default=45)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument("--max-fer-per-class", type=int, default=0)
    parser.add_argument("--max-extra-per-class", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    set_seed(SEED)

    fer_root = resolve_fer_root(base_dir, args.fer_root)
    image_paths, labels, class_counts = collect_dataset(
        fer_root=fer_root,
        extra_roots=args.extra_data,
        max_fer_per_class=args.max_fer_per_class,
        max_extra_per_class=args.max_extra_per_class,
    )

    print(f"FER root: {fer_root}")
    if args.extra_data:
        print("Extra roots:")
        for root in args.extra_data:
            print(f"  - {Path(root).expanduser().resolve()}")

    print("\nClass counts (fer + extra):")
    total_by_class = {}
    for emotion in EMOTIONS:
        fer_count = class_counts[emotion]["fer"]
        extra_count = class_counts[emotion]["extra"]
        total = fer_count + extra_count
        total_by_class[emotion] = total
        print(f"  {emotion:10s} fer={fer_count:5d} extra={extra_count:5d} total={total:5d}")

    print(f"\nTotal images: {len(image_paths)}")

    train_paths, val_paths, y_train_raw, y_val_raw = train_test_split(
        image_paths,
        labels,
        test_size=args.val_size,
        random_state=SEED,
        stratify=labels,
    )

    train_dataset = make_dataset(
        paths=train_paths,
        labels=y_train_raw,
        img_size=args.img_size,
        num_classes=len(EMOTIONS),
        batch_size=args.batch_size,
        training=True,
    )
    val_dataset = make_dataset(
        paths=val_paths,
        labels=y_val_raw,
        img_size=args.img_size,
        num_classes=len(EMOTIONS),
        batch_size=args.batch_size,
        training=False,
    )

    classes = np.unique(y_train_raw)
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_raw,
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_values)}
    print("\nClass weights:", class_weights)

    model, backbone, pretrained_loaded = build_model(
        img_size=args.img_size,
        num_classes=len(EMOTIONS),
    )

    compile_model(model, args.base_lr)

    model_path = base_dir / "emotion_model.keras"
    labels_path = base_dir / "emotion_labels.json"
    history_path = base_dir / "training_history.csv"
    summary_path = base_dir / "training_summary.json"

    callback_list = [
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
            patience=7,
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
        callbacks.CSVLogger(str(history_path), append=False),
    ]

    print("\nTraining stage 1 (frozen backbone)...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.base_epochs,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1,
    )

    if args.finetune_epochs > 0:
        print("\nTraining stage 2 (fine-tuning)...")
        unfreeze_top_layers(backbone, args.unfreeze_layers)
        compile_model(model, args.finetune_lr)

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.base_epochs + args.finetune_epochs,
            initial_epoch=args.base_epochs,
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=1,
        )

    val_metrics = model.evaluate(val_dataset, verbose=0)
    metric_names = model.metrics_names
    metric_report = {
        name: float(value)
        for name, value in zip(metric_names, val_metrics)
    }

    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(EMOTIONS, handle, indent=2)

    summary = {
        "seed": SEED,
        "fer_root": str(fer_root),
        "extra_roots": [str(Path(p).expanduser().resolve()) for p in args.extra_data],
        "total_images": len(image_paths),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "base_epochs": args.base_epochs,
        "finetune_epochs": args.finetune_epochs,
        "unfreeze_layers": args.unfreeze_layers,
        "base_lr": args.base_lr,
        "finetune_lr": args.finetune_lr,
        "pretrained_backbone_loaded": pretrained_loaded,
        "class_distribution": total_by_class,
        "class_weights": class_weights,
        "metrics": metric_report,
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "history_path": str(history_path),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTraining complete.")
    print(f"Model saved: {model_path}")
    print(f"Labels saved: {labels_path}")
    print(f"History saved: {history_path}")
    print(f"Summary saved: {summary_path}")
    print("Validation metrics:", metric_report)


if __name__ == "__main__":
    main()
