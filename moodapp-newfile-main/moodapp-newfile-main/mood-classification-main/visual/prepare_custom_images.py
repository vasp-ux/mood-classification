import argparse
import shutil
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy labeled custom emotion images into visual/custom_images/<emotion>/.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Folder with emotion subfolders (angry, happy, ...).",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Target root. Default: <this_folder>/custom_images",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional cap for copied images per class (0 = all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    source_root = Path(args.source).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    target_root = (
        Path(args.target).expanduser().resolve()
        if args.target
        else (base_dir / "custom_images")
    )
    target_root.mkdir(parents=True, exist_ok=True)

    copied_total = 0
    for emotion in EMOTIONS:
        src_dir = source_root / emotion
        dst_dir = target_root / emotion
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[warn] Missing source class folder: {src_dir}")
            continue

        files = [
            p
            for p in src_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        files.sort()

        if args.limit_per_class > 0:
            files = files[: args.limit_per_class]

        copied = 0
        for file_path in files:
            destination = dst_dir / file_path.name
            stem = destination.stem
            suffix = destination.suffix
            counter = 1
            while destination.exists():
                destination = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.copy2(file_path, destination)
            copied += 1
            copied_total += 1

        print(f"{emotion:10s} copied: {copied}")

    print(f"\nDone. Total copied images: {copied_total}")
    print(f"Target dataset: {target_root}")


if __name__ == "__main__":
    main()
