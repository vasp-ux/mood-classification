import cv2
import os
import numpy as np
from sklearn.utils import shuffle

DATASET_PATH = "/home/soorajvp/Desktop/moodclass2/visual_based/archive(1)/train_balanced"

EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "neutral","sad","surprise"]
IMG_SIZE = 48

data = []
labels = []

for label, emotion in enumerate(EMOTIONS):
    folder = os.path.join(DATASET_PATH, emotion)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            labels.append(label)

        except:
            continue

# Convert to numpy
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Add channel dimension
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Shuffle
data, labels = shuffle(data, labels, random_state=42)

# Save
np.save("X.npy", data)
np.save("y.npy", labels)

print("✅ Dataset built successfully")
print("X shape:", data.shape)
print("y shape:", labels.shape)
