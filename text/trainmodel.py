from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_txt(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ";" in line:
                text, label = line.rsplit(";", 1)
                texts.append(text.strip())
                labels.append(label.strip())
    return texts, labels

# Load dataset
X_train, y_train = load_txt("dataset/train.txt")
X_val, y_val = load_txt("dataset/val.txt")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Save
joblib.dump(model, "text_emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Text emotion model trained and saved successfully")
