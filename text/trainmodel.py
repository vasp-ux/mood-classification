import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1) Load cleaned dataset
df = pd.read_csv("/home/soorajvp/Desktop/moodclass2/text based/goemotions_clean_8.csv")

X_text = df["text"]
y = df["emotion"]

# 2) Train/validation split (for quick sanity check)
X_train, X_val, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# 3) TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 4) Train classifier (handles class imbalance)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=None
)
model.fit(X_train_vec, y_train)

# 5) Quick evaluation (optional but good)
y_pred = model.predict(X_val_vec)
print("\n=== Validation Report ===")
print(classification_report(y_val, y_pred))

# 6) Save model + vectorizer
joblib.dump(model, "text_emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ STEP 2 complete: Model and vectorizer saved")
