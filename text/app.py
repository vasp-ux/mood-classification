from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("text_emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").strip()

    if len(text) < 3:
        return jsonify({"error": "Text too short"})

    X = vectorizer.transform([text])
    emotion = model.predict(X)[0]

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
