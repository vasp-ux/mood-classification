import os
import random
import shutil

DATASET_PATH = r"C:\Users\User\Downloads\mood-classification\visual\fer2013\train"
BALANCED_PATH = r"C:\Users\User\Downloads\mood-classification\visual\fer2013\train_balanced"

EMOTIONS = [
 "angry","disgust","fear",
 "happy","neutral","sad","surprise","contempt"
]

os.makedirs(BALANCED_PATH, exist_ok=True)

# Step 1: find minimum count
counts = {}
for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)
    counts[emotion] = len(os.listdir(folder))

min_count = min(counts.values())
print("Minimum images per class:", min_count)

# Step 2: copy balanced images
for emotion in EMOTIONS:
    src = os.path.join(DATASET_PATH, emotion)
    dst = os.path.join(BALANCED_PATH, emotion)
    os.makedirs(dst, exist_ok=True)

    images = os.listdir(src)
    selected = random.sample(images, min_count)

    for img in selected:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    print(f"{emotion} balanced to {min_count} images")

print("\n✅ Dataset balancing completed")
