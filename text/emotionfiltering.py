import pandas as pd

# Load dataset
df = pd.read_csv("/home/soorajvp/Desktop/moodclass2/text based")

# Use only training split
df = df[df["split"] == "train"]

# ID → Emotion mapping (FINAL)
id_to_emotion = {
    17: "happy", 0: "happy", 5: "happy", 20: "happy", 22: "happy", 21: "happy",
    25: "sad", 11: "sad", 9: "sad", 23: "sad",
    2: "angry", 1: "angry",
    7: "disgust",
    13: "fear", 3: "fear", 19: "fear",
    26: "surprise",
    6: "contempt",
    27: "neutral"
}

texts = []
emotions = []

for _, row in df.iterrows():
    # labels look like "[27]" or "[13,19]"
    label_str = str(row["labels"])

    # remove square brackets
    label_str = label_str.replace("[", "").replace("]", "")

    label_list = label_str.split(",")

    # keep only single-label samples
    if len(label_list) == 1:
        try:
            label_id = int(label_list[0].strip())
        except ValueError:
            continue

        if label_id in id_to_emotion:
            texts.append(row["text"])
            emotions.append(id_to_emotion[label_id])

clean_df = pd.DataFrame({
    "text": texts,
    "emotion": emotions
})

clean_df.to_csv("goemotions_clean_8.csv", index=False)

print("✅ Dataset fixed and saved")
print("Total samples:", len(clean_df))
print(clean_df["emotion"].value_counts())
