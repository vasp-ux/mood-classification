from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/predict/*": {
            "origins": "*",
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "text_emotion_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/predict/text", methods=["POST"])
def predict_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()

    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    X = vectorizer.transform([text])
    emotion = model.predict(X)[0]

    return jsonify({"emotion": emotion})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
