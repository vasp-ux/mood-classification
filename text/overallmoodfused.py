import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

TEXT_WEIGHT = 0.7
VISUAL_WEIGHT = 0.3

# ✅ Correct paths
text_df = pd.read_csv("mood_diary.csv")
visual_df = pd.read_csv("../visual/mood_log.csv")

# Convert DateTime
text_df["DateTime"] = pd.to_datetime(text_df["DateTime"])
visual_df["DateTime"] = pd.to_datetime(visual_df["DateTime"])

print("\nSelect analysis period:")
print("1 → Today")
print("2 → Last 7 days")

choice = input("Enter choice (1/2): ").strip()

if choice == "1":
    start_time = datetime.now().replace(hour=0, minute=0, second=0)
    period_label = "Today"
else:
    start_time = datetime.now() - timedelta(days=7)
    period_label = "Last 7 Days"

# Filter by period
text_period = text_df[text_df["DateTime"] >= start_time]
visual_period = visual_df[visual_df["DateTime"] >= start_time]

# Count emotions
text_counts = Counter(text_period["Emotion"])
visual_counts = Counter(visual_period["Emotion"])

# Fusion
fusion_scores = {}
all_emotions = set(text_counts.keys()) | set(visual_counts.keys())

for emotion in all_emotions:
    fusion_scores[emotion] = (
        TEXT_WEIGHT * text_counts.get(emotion, 0)
        + VISUAL_WEIGHT * visual_counts.get(emotion, 0)
    )

final_mood = max(fusion_scores, key=fusion_scores.get)

print("\n🧾 FUSED MOOD SUMMARY")
print("-" * 30)
print("Period:", period_label)

print("\nText emotion counts:")
for e, c in text_counts.items():
    print(f"{e:<10}: {c}")

print("\nVisual emotion counts:")
for e, c in visual_counts.items():
    print(f"{e:<10}: {c}")

print("\n🧠 Overall Mood (Text + Visual):", final_mood)
print("\n⚠️ This system supports well-being, not medical diagnosis.")
