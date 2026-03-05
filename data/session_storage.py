import csv
import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai

# ================== ENV LOAD ==================
load_dotenv()

# ================== GEMINI SETUP ==================
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# Toggle for demo / fallback
USE_GEMINI = False
   # set False to use mock response

# ================== PATH SETUP ==================

DATA_DIR = "data"
TEXT_FILE = os.path.join(DATA_DIR, "text_sessions.csv")
VISUAL_FILE = os.path.join(DATA_DIR, "visual_sessions.csv")
OVERALL_FILE = os.path.join(DATA_DIR, "overall_sessions.csv")

# ================== INITIALIZATION ==================

def initialize_storage():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["session_id", "mood", "confidence", "timestamp"]
            )

    if not os.path.exists(VISUAL_FILE):
        with open(VISUAL_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["session_id", "mood", "confidence", "timestamp"]
            )

    if not os.path.exists(OVERALL_FILE):
        with open(OVERALL_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"]
            )

# ================== SESSION ID HANDLER ==================

def get_next_session_id(file_path):
    try:
        with open(file_path, "r") as f:
            return len(list(csv.reader(f)))
    except FileNotFoundError:
        return 1

# ================== SAVE TEXT SESSION ==================

def save_text_session(mood, confidence):
    initialize_storage()
    session_id = get_next_session_id(TEXT_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(TEXT_FILE, "a", newline="") as f:
        csv.writer(f).writerow([session_id, mood, confidence, timestamp])

    return session_id

# ================== SAVE VISUAL SESSION ==================

def save_visual_session(mood, confidence):
    initialize_storage()
    session_id = get_next_session_id(VISUAL_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(VISUAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([session_id, mood, confidence, timestamp])

    return session_id

# ================== READ LATEST SESSION ==================

def get_latest_session(file_path):
    with open(file_path, "r") as f:
        rows = list(csv.DictReader(f))
        return rows[-1] if rows else None

# ================== FUSION LOGIC ==================

def fuse_emotions(text_result, visual_result):
    text_conf = float(text_result["confidence"])
    visual_conf = float(visual_result["confidence"])

    overall_mood = (
        text_result["mood"]
        if text_conf >= visual_conf
        else visual_result["mood"]
    )

    overall_confidence = round((text_conf + visual_conf) / 2, 2)
    return overall_mood, overall_confidence

# ================== SEVERITY LOGIC ==================

def determine_severity(mood, confidence):
    if mood in ["sad", "anxious"] and confidence >= 0.8:
        return "high"
    elif mood in ["sad", "anxious"] and confidence >= 0.6:
        return "medium"
    return "low"

# ================== SAVE OVERALL SESSION ==================

def save_overall_session():
    initialize_storage()

    text_result = get_latest_session(TEXT_FILE)
    visual_result = get_latest_session(VISUAL_FILE)

    if not text_result or not visual_result:
        return None

    overall_mood, overall_confidence = fuse_emotions(text_result, visual_result)
    severity = determine_severity(overall_mood, overall_confidence)

    session_id = get_next_session_id(OVERALL_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(OVERALL_FILE, "a", newline="") as f:
        csv.writer(f).writerow(
            [session_id, overall_mood, overall_confidence, severity, timestamp]
        )

    return {
        "session_id": session_id,
        "overall_mood": overall_mood,
        "overall_confidence": overall_confidence,
        "severity": severity
    }

# ================== TREND ANALYSIS ==================

def get_last_n_overall_sessions(n=3):
    with open(OVERALL_FILE, "r") as f:
        rows = list(csv.DictReader(f))
        return rows[-n:] if len(rows) >= n else rows

def analyze_trend(n=3):
    sessions = get_last_n_overall_sessions(n)
    if not sessions:
        return "neutral"

    negative = sum(
        1 for s in sessions if s["overall_mood"] in ["sad", "anxious", "angry"]
    )
    positive = sum(
        1 for s in sessions if s["overall_mood"] == "happy"
    )

    if negative >= 2:
        return "negative"
    elif positive >= 2:
        return "positive"
    return "neutral"

# ================== SUGGESTION ENGINE ==================

def get_support_actions(mood):
    return {
        "sad": [
            "Listen to calming music",
            "Try deep breathing",
            "Write your thoughts in a journal"
        ],
        "anxious": [
            "Practice 4-7-8 breathing",
            "Grounding exercise",
            "Listen to soft music"
        ],
        "angry": [
            "Take a short walk",
            "Muscle relaxation",
            "Listen to slow music"
        ],
        "neutral": [
            "Light mindfulness",
            "Relaxing background music"
        ],
        "happy": [
            "Gratitude journaling",
            "Share positivity with a friend"
        ]
    }.get(mood, [])

def generate_suggestions(overall, trend):
    suggestions = []
    mood = overall["overall_mood"]
    severity = overall["severity"]

    suggestions.extend(get_support_actions(mood))

    if severity == "medium":
        suggestions.append("Call a trusted friend or family member")
    if severity == "high":
        suggestions.append("Contact emergency support or helpline")
    if trend == "negative" and severity != "high":
        suggestions.append("Consider talking to someone you trust")

    return list(dict.fromkeys(suggestions))

# ================== LLM PROMPT ==================

def build_llm_prompt(overall, trend, suggestions):
    return f"""
You are a calm, supportive assistant.

Overall mood: {overall['overall_mood']}
Confidence level: {overall['overall_confidence']}
Severity: {overall['severity']}
Recent emotional trend: {trend}

Suggested actions:
{', '.join(suggestions)}

Explain gently and encouragingly.
Do NOT provide medical advice.
"""

# ================== GEMINI / FALLBACK ==================

def get_llm_response(prompt):
    if not USE_GEMINI or not GEMINI_AVAILABLE:
        return (
            "Based on recent emotional patterns, you seem to be feeling low. "
            "Listening to calming music or practicing breathing exercises may help. "
            "Reaching out to someone you trust can provide emotional support."
        )

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


# ================== TEST BLOCK ==================

if __name__ == "__main__":
    print("Testing Phase 2 → 6...")

    save_text_session("sad", 0.72)
    save_visual_session("neutral", 0.65)

    overall = save_overall_session()
    print("Overall session:", overall)

    trend = analyze_trend()
    print("Trend:", trend)

    suggestions = generate_suggestions(overall, trend)
    print("Suggestions:", suggestions)

    prompt = build_llm_prompt(overall, trend, suggestions)
    llm_message = get_llm_response(prompt)

    print("\n🤖 LLM Response:\n")
    print(llm_message)
