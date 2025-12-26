from datetime import datetime
import cv2
import numpy as np
import time
import csv
from collections import Counter
from tensorflow.keras.models import load_model

# ================= CONFIG ================= #

MODEL_PATH = "/home/soorajvp/Desktop/moodclass2/emotion_model.h5"
CASCADE_PATH = "/home/soorajvp/Desktop/moodclass2/haarcascade_frontalface_default.xml"
LOG_FILE = "mood_log.csv"

EMOTIONS = [
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

IMG_SIZE = 48
NUM_CLASSES = 8



# ========================================== #

# Load model
model = load_model(MODEL_PATH)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ Haar cascade not loaded")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("📷 Webcam started (press Q to quit)")

# Session tracking
session_emotions = []
start_time = time.time()
session_start_datetime = datetime.now()
last_record_time = 0

# Face memory (ANTI-FLICKER)
last_face = None
last_emotion = "neutral"   # fallback emotion

# ================= MAIN LOOP ================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    # Update face if detected
    if len(faces) > 0:
        last_face = faces[0]

    if last_face is not None:
        x, y, w, h = last_face

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        prediction = model.predict(face, verbose=0)
        emotion = EMOTIONS[np.argmax(prediction)]
        last_emotion = emotion

        # Draw box & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame, emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0,255,0), 2
        )
    else:
        emotion = last_emotion  # reuse last emotion

    # 🔴 LOG EVERY SECOND NO MATTER WHAT
    current_time = time.time()
    if current_time - last_record_time >= 1:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_emotions.append((timestamp, emotion))
        last_record_time = current_time

    cv2.imshow("Real-Time Emotion Detection", frame)
    


    key = cv2.waitKey(1)

    # Press Q or Esc to quit
    if key == ord('q') or key == ord('Q') or key == 27:
       break


# ================= CLEANUP ================= #

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
duration = int(end_time - start_time)

# ================= SUMMARY CALCULATION ================= #

# Extract only emotions from (timestamp, emotion)
emotion_only = [emo for _, emo in session_emotions]

count = Counter(emotion_only)
total = sum(count.values())

summary_percent = {}
for emo in EMOTIONS:
    summary_percent[emo] = (count.get(emo, 0) / total) * 100

dominant_mood = count.most_common(1)[0][0]


# ================= SAVE LOG ================= #

with open(LOG_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)

    # ===== SESSION METADATA =====
    writer.writerow(["SESSION SUMMARY"])
    writer.writerow(["Session Start", session_start_datetime.strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Session End", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Session Duration (seconds)", duration])
    writer.writerow([])

    # ===== SUMMARY PERCENTAGES =====
    writer.writerow(["Emotion", "Percentage"])
    for emo in EMOTIONS:
        writer.writerow([emo, f"{summary_percent[emo]:.2f}%"])

    writer.writerow([])
    writer.writerow(["Dominant Mood", dominant_mood])
    writer.writerow([])

    # ===== PER-SECOND LOG =====
    writer.writerow(["Timestamp", "Emotion"])
    for timestamp, emo in session_emotions:
        writer.writerow([timestamp, emo])



# ================= SUMMARY ================= #

print("\n🧾 SESSION SUMMARY")
print("--------------------------")
print(f"⏱ Session duration: {duration} seconds")

count = Counter([emo for _, emo in session_emotions])
total = sum(count.values())

for emo in EMOTIONS:
    percent = (count.get(emo, 0) / total) * 100
    print(f"{emo:10s}: {percent:.2f}%")

dominant = count.most_common(1)[0][0]
print("\n🧠 Dominant Mood:", dominant)
print("📁 Mood log saved as:", LOG_FILE)
print("✅ Session completed successfully")
