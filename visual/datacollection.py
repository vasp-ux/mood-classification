import cv2
import os
import numpy as np

# Dataset path
DATASET_PATH = "/home/soorajvp/Desktop/moodclass2/visual_based/archive(1)/train"
print("Dataset exists:", os.path.exists(DATASET_PATH))

# Emotion labels (must match folder names)
EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "neutral","sad","surprise"]

IMG_SIZE = 48   # 48x48 for CNN

data = []
labels = []

for label, emotion in enumerate(EMOTIONS):
    emotion_path = os.path.join(DATASET_PATH, emotion)

    if not os.path.exists(emotion_path):
        print(f"❌ Folder not found: {emotion_path}")
        continue

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
