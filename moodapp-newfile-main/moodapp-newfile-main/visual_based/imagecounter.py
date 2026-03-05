import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "fer2013", "train")

EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

print("Dataset exists:", os.path.exists(DATASET_PATH))
print("\nImage count per emotion:\n")

total = 0
for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)

    if not os.path.isdir(folder):
        print(f"{emotion:10s} : folder missing")
        continue

    count = len(os.listdir(folder))
    total += count
    print(f"{emotion:10s} : {count}")

print("\nTotal images:", total)
