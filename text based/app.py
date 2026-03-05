from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib


app = Flask(__name__)

CORS(app, resources={
    r"/predict/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})


# Load model & vectorizer (already trained by you)
BASE_DIR = "/home/soorajvp/Desktop/moodclass2/text based"

model = joblib.load(f"{BASE_DIR}/text_emotion_model.pkl")
vectorizer = joblib.load(f"{BASE_DIR}/tfidf_vectorizer.pkl")


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

    return jsonify({
        "emotion": emotion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

