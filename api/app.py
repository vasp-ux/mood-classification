from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)   # ✅ ENABLE CORS

# Load model and vectorizer
model = joblib.load("../text/text_emotion_model.pkl")
vectorizer = joblib.load("../text/vectorizer.pkl")

@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    emotion = model.predict(X)[0]

    return jsonify({
        "emotion": emotion
    })

if __name__ == "__main__":
    app.run(debug=True)
