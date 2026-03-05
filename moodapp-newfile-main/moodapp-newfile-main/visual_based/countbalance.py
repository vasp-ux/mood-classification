import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "fer2013", "train")
BALANCED_PATH = os.path.join(BASE_DIR, "fer2013", "train_balanced")

EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

os.makedirs(BALANCED_PATH, exist_ok=True)

# Step 1: find minimum image count from available classes
counts = {}
for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(folder):
        continue
    counts[emotion] = len(os.listdir(folder))

if not counts:
    raise FileNotFoundError(f"No emotion folders found in {DATASET_PATH}")

min_count = min(counts.values())
print("Minimum images per class:", min_count)

# Step 2: copy equal number of images
for emotion in counts:
    src = os.path.join(DATASET_PATH, emotion)
    dst = os.path.join(BALANCED_PATH, emotion)
    os.makedirs(dst, exist_ok=True)

    images = os.listdir(src)
    selected = random.sample(images, min_count)

    for img in selected:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    print(f"{emotion} balanced to {min_count} images")

print("\nDataset balancing completed")
