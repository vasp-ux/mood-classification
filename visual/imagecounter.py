import os

DATASET_PATH = r"C:\Users\User\Downloads\mood-classification\visual\fer2013\train"

EMOTIONS = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

print("Dataset exists:", os.path.exists(DATASET_PATH))
print("\n📊 Image count per emotion:\n")

total = 0
for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)

    if not os.path.exists(folder):
        print(f"{emotion:10s} : ❌ folder not found")
        continue

    count = len(os.listdir(folder))
    total += count
    print(f"{emotion:10s} : {count}")

print("\nTotal images:", total)
