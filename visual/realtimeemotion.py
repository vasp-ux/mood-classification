import cv2
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# ================= CONFIG ================= #

MODEL_PATH = r"C:\Users\User\Downloads\mood-classification\visual\emotion_model.keras"
LOG_FILE = "mood_log.csv"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt"
]

IMG_SIZE = 48

# ========================================== #

model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("📷 Webcam started (press Q to quit)")

last_emotion = "neutral"
last_record_time = 0

# ================= MAIN LOOP ================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )

    emotion = last_emotion

    if len(faces) > 0:
        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        probs = model.predict(face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(probs)]
        last_emotion = emotion

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ✅ LOG EVERY 1 SECOND
    if time.time() - last_record_time >= 1:
        entry = {
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Emotion": emotion
        }

        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        else:
            df = pd.DataFrame([entry])

        df.to_csv(LOG_FILE, index=False)
        last_record_time = time.time()

    cv2.imshow("Real-Time Emotion Detection", frame)

    key = cv2.waitKey(1)
    if key in [ord('q'), ord('Q'), 27]:
        break

# ================= CLEANUP ================= #

cap.release()
cv2.destroyAllWindows()

print("📁 Visual mood log saved:", LOG_FILE)
print("✅ Session completed successfully")
