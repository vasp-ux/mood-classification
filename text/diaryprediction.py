import joblib
import pandas as pd
from datetime import datetime
import os

# Load model
model = joblib.load("text_emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input
user_text = input("\n✍️ Write your diary entry:\n").strip()

if len(user_text) < 3:
    print("❌ Please enter a meaningful diary entry.")
    exit()

# Predict
X = vectorizer.transform([user_text])
emotion = model.predict(X)[0]

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

entry = {
    "DateTime": timestamp,
    "Text": user_text,
    "Emotion": emotion
}

diary_file = "mood_diary.csv"

if os.path.exists(diary_file):
    df = pd.read_csv(diary_file)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
else:
    df = pd.DataFrame([entry])

df.to_csv(diary_file, index=False)

print("\n🧠 Detected Emotion:", emotion)
print("📅 Saved at:", timestamp)
