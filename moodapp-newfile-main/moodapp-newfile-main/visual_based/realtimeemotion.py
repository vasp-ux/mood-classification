import cv2
import json
import numpy as np
import time
import os
import csv
from collections import Counter
from datetime import datetime
import tensorflow as tf

# ================= PATHS ================= #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "emotion_labels.json")

# ================= LOAD MODEL ================= #

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
MODEL_INPUT_CHANNELS = model.input_shape[-1]
print("Model loaded successfully")

# ================= CONFIG ================= #

DEFAULT_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt",
]


def load_emotions():
    expected = int(model.output_shape[-1])
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if (
                isinstance(labels, list)
                and len(labels) == expected
                and all(isinstance(item, str) for item in labels)
            ):
                return [item.strip().lower() for item in labels]
        except Exception:
            pass
    return DEFAULT_EMOTIONS[:expected]


EMOTIONS = load_emotions()

IMG_SIZE = 48

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= GLOBALS ================= #

running = False
cap = None
session_emotions = []
per_second_log = []

start_time = None
last_record_time = 0
last_emotion = "neutral"


# ================= HELPERS ================= #

def preprocess_face(face_gray):
    face_gray = cv2.equalizeHist(face_gray)
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

    if MODEL_INPUT_CHANNELS == 3:
        face = np.stack([face, face, face], axis=-1)
    else:
        face = np.expand_dims(face, axis=-1)

    return np.expand_dims(face, axis=0)


# ================= FUNCTIONS ================= #

def start_emotion_session():
    global running, cap, session_emotions, per_second_log
    global start_time, last_record_time, last_emotion

    running = True
    session_emotions = []
    per_second_log = []

    last_emotion = "neutral"
    last_record_time = 0
    start_time = time.time()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        running = False
        print("Webcam not accessible")
        return {
            "duration": 0,
            "dominant_mood": "neutral",
            "distribution": {},
        }

    print("Webcam session started (Press Q to quit)")

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
        )

        emotion = last_emotion
        confidence = 0.0

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            face = gray[y:y + h, x:x + w]
            face = preprocess_face(face)

            probs = model.predict(face, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(probs)]
            confidence = float(np.max(probs))
            last_emotion = emotion

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ===== LEVEL-2 PER-SECOND LOGGING ===== #
        if time.time() - last_record_time >= 1:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            second = int(time.time() - start_time)

            session_emotions.append(emotion)
            per_second_log.append([
                timestamp,
                second,
                emotion,
                round(confidence, 3),
            ])

            last_record_time = time.time()

        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    summary = stop_emotion_session()
    save_session_to_csv(summary)
    save_detailed_log()

    print("\nSESSION SUMMARY")
    print("--------------------------")
    print(f"Duration: {summary['duration']} seconds")
    print(f"Dominant Mood: {summary['dominant_mood']}")
    for k, v in summary["distribution"].items():
        print(f"{k:10s}: {v:.2f}%")

    return summary


# ================= STOP SESSION ================= #

def stop_emotion_session():
    global running, cap, session_emotions, start_time

    running = False

    if cap:
        cap.release()
        cap = None
    cv2.destroyAllWindows()

    if start_time is None:
        duration = 0
    else:
        duration = int(time.time() - start_time)

    if not session_emotions:
        return {
            "duration": duration,
            "dominant_mood": "neutral",
            "distribution": {},
        }

    count = Counter(session_emotions)
    total = sum(count.values())

    distribution = {
        emo: round((count.get(emo, 0) / total) * 100, 2)
        for emo in EMOTIONS
    }

    dominant_mood = count.most_common(1)[0][0]

    return {
        "duration": duration,
        "dominant_mood": dominant_mood,
        "distribution": distribution,
    }


# ================= CSV LOGGING ================= #

def save_session_to_csv(summary):
    log_file = os.path.join(BASE_DIR, "mood_log.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Date",
                "Start Time",
                "Duration (sec)",
                "Dominant Mood",
                "Angry %",
                "Contempt %",
                "Disgust %",
                "Fear %",
                "Happy %",
                "Neutral %",
                "Sad %",
                "Surprise %",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            summary["duration"],
            summary["dominant_mood"],
            summary["distribution"].get("angry", 0),
            summary["distribution"].get("contempt", 0),
            summary["distribution"].get("disgust", 0),
            summary["distribution"].get("fear", 0),
            summary["distribution"].get("happy", 0),
            summary["distribution"].get("neutral", 0),
            summary["distribution"].get("sad", 0),
            summary["distribution"].get("surprise", 0),
        ])

    print("Session summary saved to mood_log.csv")


def save_detailed_log():
    log_file = os.path.join(BASE_DIR, "mood_log_detailed.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Second",
                "Emotion",
                "Confidence",
            ])

        for row in per_second_log:
            writer.writerow(row)

    print("Per-second emotion log saved to mood_log_detailed.csv")


# ================= MAIN ================= #

if __name__ == "__main__":
    start_emotion_session()
