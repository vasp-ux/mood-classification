import os
import sys
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd

TEXT_WEIGHT = 0.5
VISUAL_WEIGHT = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_LOG_PATH = os.path.join(BASE_DIR, "mood_diary.csv")
VISUAL_LOG_PATH = os.path.join(BASE_DIR, "..", "visual_based", "mood_log.csv")

if not os.path.exists(TEXT_LOG_PATH) or not os.path.exists(VISUAL_LOG_PATH):
    print("Required CSV file missing")
    sys.exit(1)


def load_text_counts(start_time):
    text_df = pd.read_csv(TEXT_LOG_PATH)
    text_df["DateTime"] = pd.to_datetime(text_df["DateTime"], errors="coerce")
    text_df = text_df.dropna(subset=["DateTime"])
    text_period = text_df[text_df["DateTime"] >= start_time]

    return Counter(text_period["Emotion"])


def load_visual_counts(start_time):
    visual_df = pd.read_csv(VISUAL_LOG_PATH)

    # Current visual logger writes summary rows with "Dominant Mood".
    if "Dominant Mood" in visual_df.columns:
        visual_df["DateTime"] = pd.to_datetime(
            visual_df["Date"].astype(str) + " " + visual_df["Start Time"].astype(str),
            errors="coerce",
        )
        visual_df = visual_df.dropna(subset=["DateTime"])
        visual_period = visual_df[visual_df["DateTime"] >= start_time]
        return Counter(visual_period["Dominant Mood"])

    # Fallback for logs that store an "Emotion" column directly.
    if "Emotion" in visual_df.columns:
        timestamp_col = "Timestamp" if "Timestamp" in visual_df.columns else None
        if timestamp_col:
            visual_df["DateTime"] = pd.to_datetime(visual_df[timestamp_col], errors="coerce")
            visual_df = visual_df.dropna(subset=["DateTime"])
            visual_period = visual_df[visual_df["DateTime"] >= start_time]
        else:
            visual_period = visual_df

        return Counter(visual_period["Emotion"])

    return Counter()


print("\nSelect analysis period:")
print("1 -> Today")
print("2 -> Last 7 Days")

choice = input("Enter choice (1/2): ").strip()

if choice == "1":
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    period_label = "Today"
else:
    start_time = datetime.now() - timedelta(days=7)
    period_label = "Last 7 Days"

text_counts = load_text_counts(start_time)
visual_counts = load_visual_counts(start_time)

all_emotions = set(text_counts) | set(visual_counts)

if not all_emotions:
    print("No data for selected period")
    sys.exit(0)

fusion_scores = {
    emotion: TEXT_WEIGHT * text_counts.get(emotion, 0)
    + VISUAL_WEIGHT * visual_counts.get(emotion, 0)
    for emotion in all_emotions
}

final_mood = max(fusion_scores, key=fusion_scores.get)

print("\nFUSED MOOD SUMMARY")
print("-" * 35)
print("Period:", period_label)

print("\nText Emotion Counts:", dict(text_counts))
print("Visual Emotion Counts:", dict(visual_counts))

print("\nOverall Mood (Text + Visual):", final_mood)
print("\nThis system supports emotional well-being, not medical diagnosis.")
