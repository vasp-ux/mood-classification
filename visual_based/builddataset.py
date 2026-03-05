import cv2
import os
import numpy as np
from sklearn.utils import shuffle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FER_DIR = os.path.join(BASE_DIR, "datasets", "fer2013")

EMOTIONS = [
    "angry",
    "contempt",   # will be skipped (FER has no contempt)
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = 48
X, y = [], []

print("📦 Building FER2013-only dataset...\n")

for label, emotion in enumerate(EMOTIONS):
    folder = os.path.join(FER_DIR, emotion)

    if not os.path.exists(folder):
        print(f"⚠ {emotion} not found in FER2013, skipping")
        continue

    count = 0
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)

        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        X.append(gray)
        y.append(label)
        count += 1

    print(f"✔ {emotion}: {count}")

X = np.array(X, dtype="float32") / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

np.save(os.path.join(BASE_DIR, "X.npy"), X)
np.save(os.path.join(BASE_DIR, "y.npy"), y)

print("\n✅ FER-only dataset built")
print("X shape:", X.shape)
print("y shape:", y.shape)
