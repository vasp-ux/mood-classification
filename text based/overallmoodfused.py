import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import os
import sys

TEXT_WEIGHT = 0.5
VISUAL_WEIGHT = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_LOG_PATH = os.path.join(BASE_DIR, "mood_diary.csv")
VISUAL_LOG_PATH = os.path.join(BASE_DIR, "..", "visual_based", "mood_log.csv")

if not os.path.exists(TEXT_LOG_PATH) or not os.path.exists(VISUAL_LOG_PATH):
    print("❌ Required CSV file missing")
    sys.exit()

# ================= LOAD TEXT DATA ================= #

text_df = pd.read_csv(TEXT_LOG_PATH)
text_df["DateTime"] = pd.to_datetime(text_df["DateTime"], errors="coerce")
text_df.dropna(subset=["DateTime"], inplace=True)

# ================= LOAD VISUAL DATA (NO HEADER CSV) ================= #

visual_columns = [
    "Date", "Time", "Frame",
    "Emotion",
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

visual_df = pd.read_csv(
    VISUAL_LOG_PATH,
    header=None,
    names=visual_columns
)

# 🔥 FIX: force Date & Time to string before joining
visual_df["DateTime"] = pd.to_datetime(
    visual_df["Date"].astype(str) + " " + visual_df["Time"].astype(str),
    errors="coerce"
)

visual_df.dropna(subset=["DateTime"], inplace=True)

# ================= PERIOD SELECTION ================= #

print("\nSelect analysis period:")
print("1 → Today")
print("2 → Last 7 Days")

choice = input("Enter choice (1/2): ").strip()

if choice == "1":
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    period_label = "Today"
else:
    start_time = datetime.now() - timedelta(days=7)
    period_label = "Last 7 Days"

# ================= FILTER ================= #

text_period = text_df[text_df["DateTime"] >= start_time]
visual_period = visual_df[visual_df["DateTime"] >= start_time]

# ================= COUNTS ================= #

text_counts = Counter(text_period["Emotion"])
visual_counts = Counter(visual_period["Emotion"])

all_emotions = set(text_counts) | set(visual_counts)

if not all_emotions:
    print("⚠️ No data for selected period")
    sys.exit()

# ================= FUSION ================= #

fusion_scores = {
    emotion: TEXT_WEIGHT * text_counts.get(emotion, 0)
    + VISUAL_WEIGHT * visual_counts.get(emotion, 0)
    for emotion in all_emotions
}

final_mood = max(fusion_scores, key=fusion_scores.get)

# ================= OUTPUT ================= #

print("\n🧾 FUSED MOOD SUMMARY")
print("-" * 35)
print("Period:", period_label)

print("\n📘 Text Emotion Counts:", dict(text_counts))
print("📷 Visual Emotion Counts:", dict(visual_counts))

print("\n🧠 Overall Mood (Text + Visual):", final_mood)
print("\n⚠️ This system supports emotional well-being, not medical diagnosis.")
