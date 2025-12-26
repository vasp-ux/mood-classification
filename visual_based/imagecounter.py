import os

DATASET_PATH = "/home/soorajvp/Desktop/moodclass2/archive(1)/train"

EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "neutral","sad","surprise"]

print("Dataset exists:", os.path.exists(DATASET_PATH))
print("\n📊 Image count per emotion:\n")

total = 0
for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)
    count = len(os.listdir(folder))
    total += count
    print(f"{emotion:10s} : {count}")

print("\nTotal images:", total)

