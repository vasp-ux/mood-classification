import csv
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FER_CSV = os.path.join(BASE_DIR, "fer2013.csv")
OUT_DIR = os.path.join(BASE_DIR, "datasets", "fer2013")

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

os.makedirs(OUT_DIR, exist_ok=True)

with open(FER_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        emotion = emotion_map[int(row["emotion"])]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)

        emotion_dir = os.path.join(OUT_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        cv2.imwrite(os.path.join(emotion_dir, f"{i}.png"), pixels)

print("✅ FER2013 converted successfully")
