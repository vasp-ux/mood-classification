from datetime import datetime
import cv2
import numpy as np
import time
import csv
from collections import Counter
from tensorflow.keras.models import load_model

# ================= CONFIG ================= #

MODEL_PATH = "/home/soorajvp/Desktop/moodclass2/visual_based/emotion_model.keras"
LOG_FILE = "mood_log.csv"

EMOTIONS = [
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
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

session_emotions = []
start_time = time.time()
session_start_datetime = datetime.now()
last_record_time = 0

last_emotion = "neutral"

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

        # ✅ SAME preprocessing as training
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # ❌ NO /255

        probs = model.predict(face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(probs)]
        last_emotion = emotion

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # 🔴 LOG EVERY SECOND
    if time.time() - last_record_time >= 1:
        session_emotions.append(
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion)
        )
        last_record_time = time.time()

    cv2.imshow("Real-Time Emotion Detection", frame)

    key = cv2.waitKey(1)
    if key in [ord('q'), ord('Q'), 27]:
        break

# ================= CLEANUP ================= #

cap.release()
cv2.destroyAllWindows()

duration = int(time.time() - start_time)

count = Counter([emo for _, emo in session_emotions])
total = sum(count.values())

summary_percent = {
    emo: (count.get(emo, 0) / total) * 100 for emo in EMOTIONS
}

dominant_mood = count.most_common(1)[0][0]

with open(LOG_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Emotion", "Percentage"])
    for emo in EMOTIONS:
        writer.writerow([emo, f"{summary_percent[emo]:.2f}%"])

print("\n🧠 Dominant Mood:", dominant_mood)
print("📁 Mood log saved as:", LOG_FILE)
print("✅ Session completed successfully")
