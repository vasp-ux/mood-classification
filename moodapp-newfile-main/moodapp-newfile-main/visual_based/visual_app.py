from flask import Flask, jsonify
from flask_cors import CORS
import threading

# import your refactored visual logic
from realtimeemotion import start_emotion_session, stop_emotion_session

app = Flask(__name__)
CORS(app)

session_thread = None
session_running = False


def run_visual_session():
    """
    Runs the emotion session in a background thread
    """
    start_emotion_session()


@app.route("/visual/start", methods=["POST"])
def start_visual():
    global session_thread, session_running

    if session_running:
        return jsonify({"message": "Visual session already running"}), 400

    session_running = True
    session_thread = threading.Thread(target=run_visual_session)
    session_thread.start()

    return jsonify({"message": "Visual session started"}), 200


@app.route("/visual/stop", methods=["POST"])
def stop_visual():
    global session_running

    if not session_running:
        return jsonify({"message": "No active visual session"}), 400

    result = stop_emotion_session()
    session_running = False

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
