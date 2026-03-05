import cv2
import os
import numpy as np

# ---------------- BASE DIRECTORY ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path (auto-detects location)
DATASET_PATH = os.path.join(BASE_DIR, "fer2013", "train_balanced")

print("Dataset exists:", os.path.exists(DATASET_PATH))
print("Dataset path:", DATASET_PATH)

# Emotion labels
EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMG_SIZE = 48

data = []
labels = []

for label, emotion in enumerate(EMOTIONS):

    emotion_path = os.path.join(DATASET_PATH, emotion)

    if not os.path.exists(emotion_path):
        print(f"❌ Folder not found: {emotion_path}")
        continue

    print(f"Loading {emotion} images...")

    for img_name in os.listdir(emotion_path):

        img_path = os.path.join(emotion_path, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            labels.append(label)

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

print("✅ Data loaded")
print("Total images:", len(data))
data = np.array(data) / 255.0
labels = np.array(labels)

data = data.reshape(-1, 48, 48, 1)

np.save("X.npy", data)
np.save("y.npy", labels)

print("✅ Dataset saved as X.npy and y.npy")

