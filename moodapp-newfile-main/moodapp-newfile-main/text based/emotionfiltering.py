import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(BASE_DIR, "goemotions_uk.csv")
TARGET_PATH = os.path.join(BASE_DIR, "goemotions_clean_8.csv")

# Load dataset
df = pd.read_csv(SOURCE_PATH)

# Use only training split
df = df[df["split"] == "train"]

# ID -> Emotion mapping
id_to_emotion = {
    17: "happy", 0: "happy", 5: "happy", 20: "happy", 22: "happy", 21: "happy",
    25: "sad", 11: "sad", 9: "sad", 23: "sad",
    2: "angry", 1: "angry",
    7: "disgust",
    13: "fear", 3: "fear", 19: "fear",
    26: "surprise",
    6: "contempt",
    27: "neutral",
}

text_column = "text_uk" if "text_uk" in df.columns else "text"

texts = []
emotions = []

for _, row in df.iterrows():
    label_str = str(row["labels"]).replace("[", "").replace("]", "")
    label_list = label_str.split(",")

    if len(label_list) == 1:
        try:
            label_id = int(label_list[0].strip())
        except ValueError:
            continue

        if label_id in id_to_emotion:
            texts.append(row[text_column])
            emotions.append(id_to_emotion[label_id])

clean_df = pd.DataFrame({
    "text": texts,
    "emotion": emotions,
})

clean_df.to_csv(TARGET_PATH, index=False)

print("Dataset cleaned and saved")
print("Input:", SOURCE_PATH)
print("Output:", TARGET_PATH)
print("Total samples:", len(clean_df))
print(clean_df["emotion"].value_counts())
