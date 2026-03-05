import joblib
import pandas as pd
import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "text_emotion_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
DIARY_PATH = os.path.join(BASE_DIR, "mood_diary.csv")

# Load trained model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Take diary input from user
user_text = input("\nWrite your diary entry:\n").strip()

if len(user_text) < 3:
    print("Please enter a meaningful diary entry.")
    raise SystemExit(1)

# Vectorize input
X = vectorizer.transform([user_text])

# Predict emotion
predicted_emotion = model.predict(X)[0]

# Timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create entry
entry = {
    "DateTime": timestamp,
    "Text": user_text,
    "Emotion": predicted_emotion,
}

# Save to diary CSV
if os.path.exists(DIARY_PATH):
    diary_df = pd.read_csv(DIARY_PATH)
    diary_df = pd.concat([diary_df, pd.DataFrame([entry])], ignore_index=True)
else:
    diary_df = pd.DataFrame([entry])

diary_df.to_csv(DIARY_PATH, index=False)

# Show result
print("\nDetected Emotion:", predicted_emotion)
print("Saved at:", timestamp)
