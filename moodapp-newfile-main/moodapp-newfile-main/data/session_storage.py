import csv
import os
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
import requests

# ================== ENV LOAD ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

# ================== GEMINI SETUP (REST API) ==================
GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
genai = None  # Not using SDK

# ================== OPENROUTER SETUP ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "MoodSense").strip()
OPENROUTER_AVAILABLE = bool(OPENROUTER_API_KEY)

# ================== OLLAMA SETUP ==================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1").strip()
OLLAMA_AVAILABLE = bool(OLLAMA_BASE_URL and OLLAMA_MODEL)

# Toggle for provider routing
USE_GEMINI = os.getenv("USE_GEMINI", "true").strip().lower() == "true"
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "true").strip().lower() == "true"
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").strip().lower() == "true"
LLM_AVAILABLE = (
    (USE_OPENROUTER and OPENROUTER_AVAILABLE)
    or (USE_GEMINI and GEMINI_AVAILABLE)
    or (USE_OLLAMA and OLLAMA_AVAILABLE)
)
LAST_LLM_SOURCE = "fallback"
LAST_LLM_ERROR = ""


def _parse_retry_after_seconds(value, default_seconds):
    try:
        parsed = float(value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return default_seconds


def _check_ollama_health():
    if not (USE_OLLAMA and OLLAMA_AVAILABLE):
        return False, "ollama is disabled or not configured"

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return False, f"{exc}"

    names = set()
    for item in payload.get("models", []):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            model = str(item.get("model", "")).strip()
            if name:
                names.add(name)
            if model:
                names.add(model)

    if not names:
        return False, "no local Ollama models found"

    wanted = OLLAMA_MODEL.strip()
    accepted = {wanted}
    if ":" not in wanted:
        accepted.add(f"{wanted}:latest")

    if not any(model_name in names for model_name in accepted):
        available_sample = ", ".join(sorted(names)[:5])
        return False, f"model '{OLLAMA_MODEL}' not found. Available: {available_sample}"

    return True, ""


def is_llm_ready():
    """Return True if at least one provider is currently usable."""
    if USE_OPENROUTER and OPENROUTER_AVAILABLE:
        return True
    if USE_GEMINI and GEMINI_AVAILABLE:
        return True
    if USE_OLLAMA and OLLAMA_AVAILABLE:
        ollama_ok, _ = _check_ollama_health()
        return ollama_ok
    return False

# ================== PATH SETUP ==================

DATA_DIR = CURRENT_DIR
TEXT_FILE = os.path.join(DATA_DIR, "text_sessions.csv")
VISUAL_FILE = os.path.join(DATA_DIR, "visual_sessions.csv")
OVERALL_FILE = os.path.join(DATA_DIR, "overall_sessions.csv")
WEEKLY_FILE = os.path.join(DATA_DIR, "weekly_overview.csv")
RESPONSES_FILE = os.path.join(DATA_DIR, "admin_responses.csv")
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
ACTIVITY_FILE = os.path.join(DATA_DIR, "activity_log.csv")


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

    if not os.path.exists(WEEKLY_FILE):
        with open(WEEKLY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "week_start",
                    "week_end",
                    "checkins",
                    "avg_confidence",
                    "dominant_mood",
                    "heavy_count",
                    "supportive_count",
                    "saved_at",
                ]
            )

    if not os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "type", "input", "response", "source", "meta"]
            )

    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["email", "name", "provider", "created_at", "first_login", "last_login", "logins"]
            )

    if not os.path.exists(ACTIVITY_FILE):
        with open(ACTIVITY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "email", "event_type", "mood", "confidence", "detail"]
            )


def read_csv_rows(file_path, limit=None):
    """Read a CSV file into a list of dicts (optionally limited to last N rows)."""
    initialize_storage()
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except FileNotFoundError:
        return []

    if limit:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = None
    if limit and limit > 0:
        rows = rows[-limit:]
    return rows


def log_response(response_type, input_text, response_text, source, meta=None):
    """Persist a response record for admin review."""
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_meta = ""
    if meta is not None:
        clean_meta = str(meta)

    with open(RESPONSES_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                timestamp,
                str(response_type or ""),
                str(input_text or ""),
                str(response_text or ""),
                str(source or ""),
                clean_meta,
            ]
        )


def read_responses(limit=200):
    return read_csv_rows(RESPONSES_FILE, limit=limit)


def upsert_user(email, name="", provider="", created_at=""):
    """Add/update a user in users.csv and increment login count."""
    initialize_storage()
    email = str(email or "").strip().lower()
    if not email:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    found = False

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        rows = []

    for row in rows:
        if str(row.get("email", "")).strip().lower() == email:
            found = True
            row["name"] = name or row.get("name", "")
            row["provider"] = provider or row.get("provider", "")
            row["created_at"] = created_at or row.get("created_at", "")
            row["last_login"] = now
            try:
                row["logins"] = str(int(row.get("logins", "0")) + 1)
            except (TypeError, ValueError):
                row["logins"] = "1"
            break

    if not found:
        rows.append(
            {
                "email": email,
                "name": name or "",
                "provider": provider or "",
                "created_at": created_at or "",
                "first_login": now,
                "last_login": now,
                "logins": "1",
            }
        )

    with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "email",
                "name",
                "provider",
                "created_at",
                "first_login",
                "last_login",
                "logins",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return True


def read_users(limit=500):
    return read_csv_rows(USERS_FILE, limit=limit)


def log_activity(email, event_type, mood="", confidence="", detail=""):
    """Persist an activity event for admin timeline."""
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ACTIVITY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                timestamp,
                str(email or "").strip().lower(),
                str(event_type or ""),
                str(mood or ""),
                str(confidence or ""),
                str(detail or ""),
            ]
        )


def read_activity(limit=500):
    return read_csv_rows(ACTIVITY_FILE, limit=limit)


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

# Weights for fusion: text analysis is more expressive, so it gets higher weight
TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4


def fuse_emotions(text_result, visual_result):
    """Weighted confidence fusion: text=60%, visual=40%."""
    text_conf = float(text_result["confidence"])
    visual_conf = float(visual_result["confidence"])

    text_score = text_conf * TEXT_WEIGHT
    visual_score = visual_conf * VISUAL_WEIGHT

    overall_mood = (
        text_result["mood"]
        if text_score >= visual_score
        else visual_result["mood"]
    )

    # Weighted average confidence (normalised so weights sum to 1)
    overall_confidence = round(text_score + visual_score, 2)
    return overall_mood, overall_confidence


# ================== SEVERITY LOGIC ==================

def determine_severity(mood, confidence):
    if mood in ["sad", "anxious"] and confidence >= 0.8:
        return "high"
    if mood in ["sad", "anxious"] and confidence >= 0.6:
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
        "severity": severity,
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
    if positive >= 2:
        return "positive"
    return "neutral"


# ================== CHART DATA AGGREGATION ==================

HEAVY_MOODS = {"sad", "fear", "angry", "disgust", "contempt", "anxious"}
SUPPORTIVE_MOODS = {"happy", "neutral", "surprise"}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_overall_sessions():
    """Return all saved overall sessions with parsed timestamps."""
    initialize_storage()
    sessions = []
    try:
        with open(OVERALL_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                timestamp = str(row.get("timestamp", "")).strip()
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue

                mood = str(row.get("overall_mood", "neutral")).strip().lower() or "neutral"
                confidence = max(0.0, min(_safe_float(row.get("overall_confidence", 0.0)), 1.0))
                severity = str(row.get("severity", "low")).strip().lower() or "low"

                sessions.append(
                    {
                        "timestamp": dt,
                        "mood": mood,
                        "confidence": confidence,
                        "severity": severity,
                    }
                )
    except FileNotFoundError:
        return []

    sessions.sort(key=lambda item: item["timestamp"])
    return sessions


def _aggregate_day(day, sessions):
    day_rows = [s for s in sessions if s["timestamp"].date() == day]
    mood_counts = {}
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    confidence_total = 0.0
    heavy_count = 0
    supportive_count = 0

    for row in day_rows:
        mood = row["mood"]
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
        confidence_total += row["confidence"]

        severity = row["severity"] if row["severity"] in severity_counts else "low"
        severity_counts[severity] += 1

        if mood in HEAVY_MOODS:
            heavy_count += 1
        if mood in SUPPORTIVE_MOODS:
            supportive_count += 1

    dominant_mood = "none"
    if mood_counts:
        dominant_mood = max(mood_counts.items(), key=lambda x: x[1])[0]

    avg_confidence = 0.0
    if day_rows:
        avg_confidence = round(confidence_total / len(day_rows), 4)

    return {
        "date": day.isoformat(),
        "label": day.strftime("%a"),
        "count": len(day_rows),
        "avg_confidence": avg_confidence,
        "dominant_mood": dominant_mood,
        "mood_counts": mood_counts,
        "severity_counts": severity_counts,
        "heavy_count": heavy_count,
        "supportive_count": supportive_count,
    }


def get_daily_chart_points(days=7, sessions=None):
    """Return daily aggregated chart points for the latest N days."""
    if sessions is None:
        sessions = get_overall_sessions()
    days = max(1, int(days))
    today = datetime.now().date()

    points = []
    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        points.append(_aggregate_day(day, sessions))
    return points


def _get_week_start(day):
    return day - timedelta(days=day.weekday())


def _aggregate_week(start_day, end_day, sessions):
    week_rows = [s for s in sessions if start_day <= s["timestamp"].date() <= end_day]
    mood_counts = {}
    confidence_total = 0.0
    heavy_count = 0
    supportive_count = 0

    for row in week_rows:
        mood = row["mood"]
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
        confidence_total += row["confidence"]
        if mood in HEAVY_MOODS:
            heavy_count += 1
        if mood in SUPPORTIVE_MOODS:
            supportive_count += 1

    dominant_mood = "none"
    if mood_counts:
        dominant_mood = max(mood_counts.items(), key=lambda x: x[1])[0]

    avg_confidence = 0.0
    if week_rows:
        avg_confidence = round(confidence_total / len(week_rows), 4)

    return {
        "week_start": start_day.isoformat(),
        "week_end": end_day.isoformat(),
        "label": f"{start_day.strftime('%b %d')} - {end_day.strftime('%b %d')}",
        "count": len(week_rows),
        "avg_confidence": avg_confidence,
        "dominant_mood": dominant_mood,
        "mood_counts": mood_counts,
        "heavy_count": heavy_count,
        "supportive_count": supportive_count,
    }


def get_weekly_chart_points(weeks=8, sessions=None):
    """Return weekly aggregated chart points for the latest N weeks."""
    if sessions is None:
        sessions = get_overall_sessions()
    weeks = max(1, int(weeks))
    today = datetime.now().date()
    current_week_start = _get_week_start(today)

    points = []
    for offset in range(weeks - 1, -1, -1):
        week_start = current_week_start - timedelta(days=7 * offset)
        week_end = week_start + timedelta(days=6)
        points.append(_aggregate_week(week_start, week_end, sessions))
    return points


def save_weekly_snapshot(weekly_points):
    """Persist weekly aggregated data for reporting/export."""
    initialize_storage()
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(WEEKLY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "week_start",
                "week_end",
                "checkins",
                "avg_confidence",
                "dominant_mood",
                "heavy_count",
                "supportive_count",
                "saved_at",
            ]
        )
        for point in weekly_points:
            writer.writerow(
                [
                    point.get("week_start", ""),
                    point.get("week_end", ""),
                    point.get("count", 0),
                    point.get("avg_confidence", 0.0),
                    point.get("dominant_mood", "none"),
                    point.get("heavy_count", 0),
                    point.get("supportive_count", 0),
                    saved_at,
                ]
            )


def get_chart_data(days=7, weeks=8):
    """Build daily + weekly chart datasets and save weekly snapshot."""
    days = max(1, int(days))
    weeks = max(1, int(weeks))

    sessions = get_overall_sessions()
    daily_points = get_daily_chart_points(days=days, sessions=sessions)
    weekly_points = get_weekly_chart_points(weeks=weeks, sessions=sessions)
    save_weekly_snapshot(weekly_points)

    total_checkins = sum(point["count"] for point in daily_points)
    confidence_weighted_sum = sum(point["avg_confidence"] * point["count"] for point in daily_points)
    avg_confidence = 0.0
    if total_checkins > 0:
        avg_confidence = round(confidence_weighted_sum / total_checkins, 4)

    return {
        "daily": daily_points,
        "weekly": weekly_points,
        "saved_weekly_file": WEEKLY_FILE,
        "stats": {
            "daily_checkins": total_checkins,
            "daily_avg_confidence": avg_confidence,
        },
    }


# ================== SUGGESTION ENGINE ==================

def get_support_actions(mood):
    return {
        "sad": [
            "Listen to calming music",
            "Try deep breathing",
            "Write your thoughts in a journal",
        ],
        "anxious": [
            "Practice 4-7-8 breathing",
            "Grounding exercise",
            "Listen to soft music",
        ],
        "angry": [
            "Take a short walk",
            "Muscle relaxation",
            "Listen to slow music",
        ],
        "neutral": [
            "Light mindfulness",
            "Relaxing background music",
        ],
        "happy": [
            "Gratitude journaling",
            "Share positivity with a friend",
        ],
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

def _get_openrouter_response(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a calm, supportive wellbeing assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 220,
        "temperature": 0.6,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)

    # Some free models reject system/developer-style instructions.
    if resp.status_code == 400:
        try:
            error_payload = resp.json()
            error_msg = str(error_payload.get("error", {}).get("message", ""))
            raw_msg = str(
                error_payload.get("error", {})
                .get("metadata", {})
                .get("raw", "")
            )
            combined_msg = f"{error_msg}\n{raw_msg}".lower()
        except Exception:
            combined_msg = str(resp.text).lower()

        if "developer instruction is not enabled" in combined_msg:
            fallback_payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "You are a calm, supportive wellbeing assistant.\n\n"
                            f"{prompt}"
                        ),
                    }
                ],
                "max_tokens": 220,
                "temperature": 0.6,
            }
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=fallback_payload,
                timeout=45,
            )

    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        # Some providers return segmented content.
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p]).strip() or None

    if isinstance(content, str):
        return content.strip() or None

    return None


def _get_gemini_response(prompt):
    # Transient 429s happen frequently on free tiers; short retry helps fallback quality.
    attempts = 3
    data = None

    for attempt in range(attempts):
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )

        if resp.status_code == 429 and attempt < attempts - 1:
            retry_after = _parse_retry_after_seconds(
                resp.headers.get("Retry-After"),
                default_seconds=1.5 * (attempt + 1),
            )
            time.sleep(retry_after)
            continue

        resp.raise_for_status()
        data = resp.json()
        break

    if data is None:
        return None

    candidates = data.get("candidates", [])
    if not candidates:
        return None

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

    if text_parts:
        return "\n".join(text_parts).strip()
    return None


def _get_ollama_response(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a calm, supportive wellbeing assistant."},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.6,
            "num_predict": 220,
        },
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()

    message = data.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip() or None

    return None


def get_llm_response(prompt):
    global LAST_LLM_SOURCE, LAST_LLM_ERROR
    LAST_LLM_SOURCE = "unavailable"
    LAST_LLM_ERROR = ""
    provider_errors = []

    provider_chain = []
    if USE_OPENROUTER and OPENROUTER_AVAILABLE:
        provider_chain.append(("openrouter", _get_openrouter_response))
    if USE_GEMINI and GEMINI_AVAILABLE:
        provider_chain.append(("gemini", _get_gemini_response))
    if USE_OLLAMA and OLLAMA_AVAILABLE:
        ollama_ok, ollama_error = _check_ollama_health()
        if ollama_ok:
            provider_chain.append(("ollama", _get_ollama_response))
        else:
            provider_errors.append(f"ollama: {ollama_error}")

    if not provider_chain:
        LAST_LLM_ERROR = "no provider configured"
        return "AI feedback is unavailable because no provider is configured."

    for provider_name, provider_fn in provider_chain:
        try:
            text = provider_fn(prompt)
            if text:
                LAST_LLM_SOURCE = provider_name
                return text
            provider_errors.append(f"{provider_name}: empty response")
        except Exception as e:
            provider_errors.append(f"{provider_name}: {e}")

    LAST_LLM_ERROR = " | ".join(provider_errors) if provider_errors else "provider call failed"
    return f"AI feedback unavailable after provider attempts. {LAST_LLM_ERROR}"


# ================== CHAT COMPANION ==================

chat_history = []


def _trim_chat_history(max_turns=20):
    """Keep only the most recent chat turns."""
    if len(chat_history) > max_turns:
        del chat_history[: len(chat_history) - max_turns]


def sync_reflection_to_companion(reflection_text, companion_reply):
    """
    Add reflection + companion reply to shared chat history so both
    reflection and AI chat stay in one continuous conversation.
    """
    reflection = str(reflection_text or "").strip()
    reply = str(companion_reply or "").strip()

    if reflection:
        chat_history.append({"role": "user", "content": reflection})
    if reply:
        chat_history.append({"role": "assistant", "content": reply})

    _trim_chat_history()


def get_chat_response(user_message):
    """Send a message to the LLM with recent chat context and return its reply."""
    global LAST_LLM_SOURCE, LAST_LLM_ERROR

    chat_history.append({"role": "user", "content": user_message})

    # Build a prompt with the last 5 exchanges for context
    context_lines = []
    for turn in chat_history[-10:]:  # last 5 pairs = 10 items
        prefix = "Student" if turn["role"] == "user" else "MoodSense"
        context_lines.append(f"{prefix}: {turn['content']}")

    prompt = f"""You are MoodSense, a warm and empathetic student wellbeing companion.

Recent conversation:
{chr(10).join(context_lines)}

Rules:
- Be warm, human, and conversational
- Do NOT mention AI, models, algorithms, or confidence scores
- Do NOT diagnose or give medical advice
- Keep your response under 80 words
- Use a supportive, non-judgmental tone
- If a student seems in crisis, gently suggest reaching out to a trusted person or helpline

Respond as MoodSense:"""

    reply = get_llm_response(prompt)
    chat_history.append({"role": "assistant", "content": reply})

    # Keep history manageable (last 20 turns)
    _trim_chat_history()

    return reply


def get_llm_status():
    """Return the current LLM provider status."""
    ollama_ok, ollama_error = _check_ollama_health()

    providers_priority = []
    if USE_OPENROUTER and OPENROUTER_AVAILABLE:
        providers_priority.append({"provider": "openrouter", "model": OPENROUTER_MODEL})
    if USE_GEMINI and GEMINI_AVAILABLE:
        providers_priority.append({"provider": "gemini", "model": GEMINI_MODEL})
    if USE_OLLAMA and OLLAMA_AVAILABLE and ollama_ok:
        providers_priority.append({"provider": "ollama", "model": OLLAMA_MODEL})

    available = len(providers_priority) > 0
    provider = providers_priority[0]["provider"] if available else "none"
    model = providers_priority[0]["model"] if available else ""

    return {
        "available": available,
        "provider": provider,
        "model": model,
        "providers_priority": providers_priority,
        "ai_mode": available,
        "providers": {
            "openrouter": {
                "enabled": USE_OPENROUTER,
                "configured": OPENROUTER_AVAILABLE,
                "model": OPENROUTER_MODEL,
            },
            "gemini": {
                "enabled": USE_GEMINI,
                "configured": GEMINI_AVAILABLE,
                "model": GEMINI_MODEL,
            },
            "ollama": {
                "enabled": USE_OLLAMA,
                "configured": OLLAMA_AVAILABLE,
                "ready": ollama_ok,
                "model": OLLAMA_MODEL,
                "base_url": OLLAMA_BASE_URL,
                "error": ollama_error if not ollama_ok else "",
            },
        },
    }


# ================== TEST BLOCK ==================

if __name__ == "__main__":
    print("Testing Phase 2 -> 6...")

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

    print("\nLLM Response:\n")
    print(llm_message)
