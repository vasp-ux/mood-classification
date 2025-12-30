import joblib
import pandas as pd
import datetime
import os

# Load trained model and vectorizer
model = joblib.load("text_emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Take diary input from user
user_text = input("\n✍️ Write your diary entry:\n").strip()

if len(user_text) < 3:
    print("❌ Please enter a meaningful diary entry.")
    exit()

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
    "Emotion": predicted_emotion
}

# Save to diary CSV
diary_file = "mood_diary.csv"

if os.path.exists(diary_file):
    diary_df = pd.read_csv(diary_file)
    diary_df = pd.concat([diary_df, pd.DataFrame([entry])], ignore_index=True)
else:
    diary_df = pd.DataFrame([entry])

diary_df.to_csv(diary_file, index=False)

# Show result
print("\n🧠 Detected Emotion:", predicted_emotion)
print("📅 Saved at:", timestamp)
