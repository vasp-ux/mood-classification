from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from tensorflow.keras.models import load_model
except Exception as exc:  # pragma: no cover
    load_model = None
    MODEL_IMPORT_ERROR = str(exc)
else:
    MODEL_IMPORT_ERROR = ""

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Load env from either this project root or monorepo root if present.
load_dotenv(PROJECT_DIR / ".env", override=False)
load_dotenv(PROJECT_DIR.parent / ".env", override=False)

TEXT_MODEL_PATH = PROJECT_DIR / "text" / "text_emotion_model.pkl"
VECTORIZER_PATH = PROJECT_DIR / "text" / "vectorizer.pkl"
VISUAL_MODEL_CANDIDATES = [
    PROJECT_DIR / "visual" / "emotion_model.keras",
    PROJECT_DIR / "emotion_model.keras",
    PROJECT_DIR.parent / "visual_based" / "emotion_model.keras",
]

DATA_DIR = BASE_DIR / "session_data"
TEXT_SESSION_FILE = DATA_DIR / "text_sessions.csv"
VISUAL_SESSION_FILE = DATA_DIR / "visual_sessions.csv"
FUSED_SESSION_FILE = DATA_DIR / "fused_sessions.csv"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Mood Classification").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").strip().lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

VISUAL_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt",
]

TEXT_TO_VISUAL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "fear": "fear",
    "joy": "happy",
    "love": "happy",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
    "disgust": "disgust",
    "contempt": "contempt",
    "happy": "happy",
}

HEAVY_MOODS = {"sad", "fear", "angry", "disgust", "contempt"}
STEADY_MOODS = {"happy", "neutral", "surprise"}

TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4


text_model = None
vectorizer = None
visual_model = None
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

text_model_error = ""
visual_model_error = ""
visual_model_source = ""


def _initialize_models() -> None:
    global text_model
    global vectorizer
    global visual_model
    global text_model_error
    global visual_model_error
    global visual_model_source

    try:
        text_model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as exc:
        text_model_error = str(exc)

    if load_model is None:
        visual_model_error = f"tensorflow import failed: {MODEL_IMPORT_ERROR}"
        return

    visual_errors = []
    for candidate in VISUAL_MODEL_CANDIDATES:
        if not candidate.exists():
            visual_errors.append(f"{candidate} (missing)")
            continue

        try:
            visual_model = load_model(candidate, compile=False)
            visual_model_source = str(candidate)
            visual_model_error = ""
            break
        except Exception as exc:
            visual_errors.append(f"{candidate} ({exc})")

    if visual_model is None:
        visual_model_error = "Unable to load visual model: " + " | ".join(visual_errors)


_initialize_models()


# ============================
# Storage helpers
# ============================

def _ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_csv(
        TEXT_SESSION_FILE,
        [
            "session_id",
            "text",
            "emotion",
            "normalized_emotion",
            "confidence",
            "timestamp",
        ],
    )
    _ensure_csv(
        VISUAL_SESSION_FILE,
        [
            "session_id",
            "duration_seconds",
            "dominant_emotion",
            "average_confidence",
            "emotion_changes",
            "stability_score",
            "distribution_json",
            "timeline_json",
            "timestamp",
        ],
    )
    _ensure_csv(
        FUSED_SESSION_FILE,
        [
            "session_id",
            "text_emotion",
            "visual_emotion",
            "fused_mood",
            "fused_confidence",
            "support_source",
            "support_message",
            "timestamp",
        ],
    )



def _ensure_csv(path: Path, header: list[str]) -> None:
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)



def _next_session_id(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for _ in handle)
        return max(1, row_count)
    except FileNotFoundError:
        return 1



def _append_row(path: Path, row: list[Any]) -> int:
    _ensure_storage()
    session_id = _next_session_id(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([session_id] + row)
    return session_id


# ============================
# Utility helpers
# ============================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default



def _normalize_text_emotion(label: str) -> str:
    lowered = str(label).strip().lower()
    return TEXT_TO_VISUAL_MAP.get(lowered, "neutral")



def _get_text_confidence(text_vector: Any, emotion: str) -> float:
    if hasattr(text_model, "predict_proba"):
        probabilities = text_model.predict_proba(text_vector)[0]
        class_list = list(text_model.classes_)
        if emotion in class_list:
            idx = class_list.index(emotion)
            return float(probabilities[idx])
    return 0.5



def _predict_text(text: str) -> tuple[str, float, str]:
    if text_model is None or vectorizer is None:
        raise RuntimeError(f"Text model not available: {text_model_error}")

    text_vector = vectorizer.transform([text])
    emotion = str(text_model.predict(text_vector)[0]).strip().lower()
    confidence = round(_get_text_confidence(text_vector, emotion), 4)
    normalized = _normalize_text_emotion(emotion)
    return emotion, confidence, normalized



def _predict_visual_from_frame(frame: np.ndarray, fallback_emotion: str) -> dict[str, Any]:
    if visual_model is None:
        raise RuntimeError(f"Visual model not available: {visual_model_error}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return {
            "face_detected": False,
            "emotion": fallback_emotion,
            "confidence": 0.0,
        }

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    face = gray[y : y + h, x : x + w]
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (48, 48)).astype("float32") / 255.0

    input_channels = int(visual_model.input_shape[-1])
    if input_channels == 3:
        face = np.stack([face, face, face], axis=-1)
    else:
        face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)

    probs = visual_model.predict(face, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    if top_idx >= len(VISUAL_EMOTIONS):
        top_idx = 4

    emotion = VISUAL_EMOTIONS[top_idx]
    confidence = float(np.max(probs))

    return {
        "face_detected": True,
        "emotion": emotion,
        "confidence": round(confidence, 4),
    }



def _count_emotion_changes(sequence: list[str]) -> int:
    if len(sequence) < 2:
        return 0

    changes = 0
    previous = sequence[0]
    for current in sequence[1:]:
        if current != previous:
            changes += 1
        previous = current
    return changes



def _build_distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {"neutral": 1.0}

    return {emotion: round(count / total, 4) for emotion, count in counter.items()}



def _analyze_webcam_video(duration_seconds: int, sample_rate_hz: float = 2.0) -> dict[str, Any]:
    if visual_model is None:
        raise RuntimeError(f"Visual model not available: {visual_model_error}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    timeline: list[dict[str, Any]] = []
    fallback_emotion = "neutral"

    start_time = time.time()
    next_sample_time = start_time
    sample_period = 1.0 / max(0.5, sample_rate_hz)

    while (time.time() - start_time) < duration_seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()
        if now < next_sample_time:
            continue

        result = _predict_visual_from_frame(frame, fallback_emotion)
        if result["face_detected"]:
            fallback_emotion = result["emotion"]

        timeline.append(
            {
                "second": round(now - start_time, 2),
                "emotion": result["emotion"],
                "confidence": float(result["confidence"]),
                "face_detected": bool(result["face_detected"]),
            }
        )

        next_sample_time = now + sample_period

    cap.release()

    if not timeline:
        timeline = [
            {
                "second": 0.0,
                "emotion": "neutral",
                "confidence": 0.0,
                "face_detected": False,
            }
        ]

    detected = [entry for entry in timeline if entry["face_detected"]]
    scoring_rows = detected if detected else timeline

    emotion_sequence = [str(entry["emotion"]).lower() for entry in scoring_rows]
    emotion_counter: Counter[str] = Counter(emotion_sequence)
    dominant_emotion = emotion_counter.most_common(1)[0][0]

    confidence_values = [float(entry["confidence"]) for entry in scoring_rows]
    average_confidence = round(
        sum(confidence_values) / max(1, len(confidence_values)),
        4,
    )

    emotion_changes = _count_emotion_changes(emotion_sequence)
    stability = round(1.0 - (emotion_changes / max(1, len(emotion_sequence) - 1)), 4)

    return {
        "duration_seconds": int(duration_seconds),
        "samples": len(timeline),
        "dominant_emotion": dominant_emotion,
        "average_confidence": average_confidence,
        "emotion_changes": emotion_changes,
        "stability_score": stability,
        "distribution": _build_distribution(emotion_counter),
        "timeline": timeline,
    }



def _save_text_session(text: str, emotion: str, normalized: str, confidence: float) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _append_row(
        TEXT_SESSION_FILE,
        [text, emotion, normalized, round(confidence, 4), timestamp],
    )



def _save_visual_session(visual: dict[str, Any]) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _append_row(
        VISUAL_SESSION_FILE,
        [
            int(visual.get("duration_seconds", 0)),
            visual.get("dominant_emotion", "neutral"),
            round(_safe_float(visual.get("average_confidence", 0.0)), 4),
            int(visual.get("emotion_changes", 0)),
            round(_safe_float(visual.get("stability_score", 0.0)), 4),
            json.dumps(visual.get("distribution", {}), ensure_ascii=True),
            json.dumps(visual.get("timeline", []), ensure_ascii=True),
            timestamp,
        ],
    )



def _save_fused_session(
    text_emotion: str,
    visual_emotion: str,
    fused_mood: str,
    fused_confidence: float,
    support_source: str,
    support_message: str,
) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _append_row(
        FUSED_SESSION_FILE,
        [
            text_emotion,
            visual_emotion,
            fused_mood,
            round(fused_confidence, 4),
            support_source,
            support_message,
            timestamp,
        ],
    )



def _load_latest_visual_session() -> dict[str, Any] | None:
    if not VISUAL_SESSION_FILE.exists():
        return None

    with VISUAL_SESSION_FILE.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    row = rows[-1]

    distribution = {}
    timeline = []
    try:
        distribution = json.loads(str(row.get("distribution_json", "{}")))
    except Exception:
        distribution = {}

    try:
        timeline = json.loads(str(row.get("timeline_json", "[]")))
    except Exception:
        timeline = []

    return {
        "duration_seconds": int(float(row.get("duration_seconds", 0) or 0)),
        "dominant_emotion": str(row.get("dominant_emotion", "neutral")).lower(),
        "average_confidence": round(_safe_float(row.get("average_confidence", 0.0)), 4),
        "emotion_changes": int(float(row.get("emotion_changes", 0) or 0)),
        "stability_score": round(_safe_float(row.get("stability_score", 0.0)), 4),
        "distribution": distribution,
        "timeline": timeline,
        "session_id": int(float(row.get("session_id", 0) or 0)),
    }



def _parse_visual_result(raw: dict[str, Any]) -> dict[str, Any]:
    distribution = raw.get("distribution")
    if not isinstance(distribution, dict):
        distribution = {}

    normalized_distribution = {
        str(key).lower(): round(_safe_float(value, 0.0), 4)
        for key, value in distribution.items()
    }

    return {
        "duration_seconds": int(float(raw.get("duration_seconds", 0) or 0)),
        "dominant_emotion": str(raw.get("dominant_emotion", "neutral")).lower(),
        "average_confidence": round(_safe_float(raw.get("average_confidence", 0.0)), 4),
        "emotion_changes": int(float(raw.get("emotion_changes", 0) or 0)),
        "stability_score": round(_safe_float(raw.get("stability_score", 0.0)), 4),
        "distribution": normalized_distribution,
        "timeline": raw.get("timeline") if isinstance(raw.get("timeline"), list) else [],
    }



def _fuse_text_and_visual(
    text_emotion: str,
    text_confidence: float,
    visual_emotion: str,
    visual_confidence: float,
    emotion_changes: int,
) -> tuple[str, float]:
    score_board = Counter()

    score_board[_normalize_text_emotion(text_emotion)] += text_confidence * TEXT_WEIGHT
    score_board[str(visual_emotion).lower()] += visual_confidence * VISUAL_WEIGHT

    if emotion_changes >= 4:
        score_board["neutral"] += 0.05

    fused_mood = score_board.most_common(1)[0][0] if score_board else "neutral"
    fused_confidence = round(min(1.0, sum(score_board.values())), 4)

    return fused_mood, fused_confidence



def _build_feeling_explanation(
    text_emotion: str,
    visual_emotion: str,
    fused_mood: str,
    emotion_changes: int,
) -> str:
    if emotion_changes >= 4:
        change_note = (
            "Your facial expressions changed several times during the short video, "
            "which can happen when stress and mixed feelings are present together."
        )
    elif emotion_changes >= 2:
        change_note = (
            "Your facial expressions shifted a bit during the video, suggesting your emotions "
            "may be moving between states."
        )
    else:
        change_note = (
            "Your facial expressions were relatively steady in this short video segment."
        )

    return (
        f"From your text, your emotion looked closest to {text_emotion}. "
        f"From your video, your dominant expression was {visual_emotion}. "
        f"Combined together, your current emotional direction looks closest to {fused_mood}. "
        f"{change_note}"
    )



def _recommended_actions(fused_mood: str, emotion_changes: int) -> list[str]:
    mood = str(fused_mood).lower()

    if mood in HEAVY_MOODS:
        base = [
            "Take 2 minutes of slow breathing: inhale 4 seconds, exhale 6 seconds.",
            "Write one sentence about what is hardest right now and one sentence about what you need.",
            "Reach out to one trusted person and share how you feel today.",
        ]
    elif mood in STEADY_MOODS:
        base = [
            "Keep a short reflection journal so you can track what is helping you.",
            "Take a short walk or stretch break to keep emotional balance.",
            "Do one small task now to maintain momentum and confidence.",
        ]
    else:
        base = [
            "Pause for one minute and name what you are feeling without judgment.",
            "Drink water and take a brief reset before your next task.",
            "Talk with someone you trust if the feeling stays heavy.",
        ]

    if emotion_changes >= 4:
        base.insert(
            1,
            "Because emotions changed quickly in the video, try two short check-ins today: afternoon and evening.",
        )

    return base[:3]



def _build_openrouter_prompt(
    text: str,
    text_emotion: str,
    visual_emotion: str,
    fused_mood: str,
    emotion_changes: int,
    actions: list[str],
) -> str:
    action_lines = "\n".join([f"- {action}" for action in actions])

    return f"""
You are a supportive emotional wellness assistant.

User reflection:
{text}

Signals:
- Text emotion: {text_emotion}
- Dominant facial expression from short video: {visual_emotion}
- Fused mood: {fused_mood}
- Expression changes during video: {emotion_changes}

Write a response in plain language with this structure:
1) one short validation sentence
2) one sentence helping the user understand what they may be going through
3) 2 practical steps from this list (adapt wording naturally):
{action_lines}

Rules:
- no diagnosis
- no mention of models, confidence, or AI
- under 110 words
""".strip()



def _call_openrouter(prompt: str) -> tuple[str | None, str]:
    if not OPENROUTER_API_KEY:
        return None, "fallback"

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
            {
                "role": "system",
                "content": "You are a gentle wellbeing coach.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": 220,
        "temperature": 0.6,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )

        if response.status_code == 400:
            combined_error = str(response.text).lower()
            try:
                body = response.json()
                message = str(body.get("error", {}).get("message", ""))
                raw = str(body.get("error", {}).get("metadata", {}).get("raw", ""))
                combined_error = f"{message}\n{raw}".lower()
            except Exception:
                pass

            if "developer instruction is not enabled" in combined_error:
                fallback_payload = {
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "You are a gentle wellbeing coach.\n\n"
                                f"{prompt}"
                            ),
                        }
                    ],
                    "max_tokens": 220,
                    "temperature": 0.6,
                }
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=fallback_payload,
                    timeout=45,
                )

        response.raise_for_status()
        body = response.json()

        choices = body.get("choices", [])
        if not choices:
            return None, "fallback"

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str) and content.strip():
            return content.strip(), "openrouter"

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"].strip())
            if text_parts:
                return "\n".join(text_parts).strip(), "openrouter"

    except Exception:
        return None, "fallback"

    return None, "fallback"


def _call_gemini(prompt: str) -> tuple[str | None, str]:
    if not GEMINI_API_KEY:
        return None, "fallback"

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        body = response.json()

        candidates = body.get("candidates", [])
        if not candidates:
            return None, "fallback"

        candidate_content = candidates[0].get("content", {})
        parts = candidate_content.get("parts", [])
        text_parts = []
        for part in parts:
            if isinstance(part, dict):
                text_value = part.get("text", "")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())

        if text_parts:
            return "\n".join(text_parts).strip(), "gemini"
    except Exception:
        return None, "fallback"

    return None, "fallback"


def _call_ollama(prompt: str) -> tuple[str | None, str]:
    if not OLLAMA_ENABLED:
        return None, "fallback"

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a gentle wellbeing coach.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": 0.6,
            "num_predict": 220,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        body = response.json()

        message = body.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip(), "ollama"
    except Exception:
        return None, "fallback"

    return None, "fallback"


# ============================
# API endpoints
# ============================

@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "text_model_ready": text_model is not None,
            "visual_model_ready": visual_model is not None,
            "text_model_error": text_model_error,
            "visual_model_error": visual_model_error,
            "visual_model_source": visual_model_source,
            "llm_openrouter_configured": bool(OPENROUTER_API_KEY),
            "llm_gemini_configured": bool(GEMINI_API_KEY),
            "llm_gemini_model": GEMINI_MODEL if GEMINI_API_KEY else "",
            "llm_ollama_enabled": OLLAMA_ENABLED,
            "llm_ollama_model": OLLAMA_MODEL if OLLAMA_ENABLED else "",
            "llm_ollama_base_url": OLLAMA_BASE_URL if OLLAMA_ENABLED else "",
        }
    ), 200


@app.route("/predict-text", methods=["POST"])
@app.route("/predict/text", methods=["POST"])
def predict_text():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if len(text) < 3:
        return jsonify({"error": "Please provide at least 3 characters of text."}), 400

    try:
        emotion, confidence, normalized = _predict_text(text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    save_session = bool(payload.get("save_session", True))
    session_id = None
    if save_session:
        session_id = _save_text_session(text, emotion, normalized, confidence)

    return jsonify(
        {
            "emotion": emotion,
            "normalized_emotion": normalized,
            "confidence": confidence,
            "saved": save_session,
            "session_id": session_id,
        }
    ), 200


@app.post("/visual/analyze-video")
def visual_analyze_video():
    payload = request.get_json(silent=True) or {}

    if visual_model is None:
        return jsonify({"error": f"Visual model unavailable: {visual_model_error}"}), 503

    duration_seconds = int(float(payload.get("duration_seconds", 8) or 8))
    duration_seconds = max(3, min(duration_seconds, 30))

    try:
        result = _analyze_webcam_video(duration_seconds=duration_seconds)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    save_session = bool(payload.get("save_session", True))
    session_id = None
    if save_session:
        session_id = _save_visual_session(result)

    result["saved"] = save_session
    result["session_id"] = session_id

    return jsonify(result), 200


@app.get("/visual/latest")
def visual_latest():
    latest = _load_latest_visual_session()
    if latest is None:
        return jsonify({"error": "No visual session found."}), 404
    return jsonify(latest), 200


@app.post("/support/fused")
def support_fused():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if len(text) < 3:
        return jsonify({"error": "Please provide a text reflection before fusion."}), 400

    try:
        text_emotion, text_confidence, text_normalized = _predict_text(text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    visual_raw = payload.get("visual_result")
    if isinstance(visual_raw, dict):
        visual_result = _parse_visual_result(visual_raw)
        visual_session_id = _save_visual_session(visual_result)
    else:
        visual_result = _load_latest_visual_session()
        visual_session_id = visual_result.get("session_id") if visual_result else None

    if visual_result is None:
        return jsonify(
            {
                "error": "No visual session available. Run visual analysis first.",
            }
        ), 400

    text_session_id = _save_text_session(
        text,
        text_emotion,
        text_normalized,
        text_confidence,
    )

    visual_emotion = str(visual_result.get("dominant_emotion", "neutral")).lower()
    visual_confidence = _safe_float(visual_result.get("average_confidence", 0.0), 0.0)
    emotion_changes = int(float(visual_result.get("emotion_changes", 0) or 0))

    fused_mood, fused_confidence = _fuse_text_and_visual(
        text_emotion=text_emotion,
        text_confidence=text_confidence,
        visual_emotion=visual_emotion,
        visual_confidence=visual_confidence,
        emotion_changes=emotion_changes,
    )

    feeling_explanation = _build_feeling_explanation(
        text_emotion=text_emotion,
        visual_emotion=visual_emotion,
        fused_mood=fused_mood,
        emotion_changes=emotion_changes,
    )
    actions = _recommended_actions(fused_mood, emotion_changes)

    prompt = _build_openrouter_prompt(
        text=text,
        text_emotion=text_emotion,
        visual_emotion=visual_emotion,
        fused_mood=fused_mood,
        emotion_changes=emotion_changes,
        actions=actions,
    )

    llm_message, llm_source = _call_openrouter(prompt)
    if not llm_message:
        llm_message, llm_source = _call_gemini(prompt)
    if not llm_message:
        llm_message, llm_source = _call_ollama(prompt)

    if not llm_message:
        return jsonify(
            {
                "error": "AI feedback unavailable. Configure at least one provider: OpenRouter, Gemini, or Ollama.",
                "providers_checked": ["openrouter", "gemini", "ollama"],
                "text_emotion": text_emotion,
                "visual_dominant_emotion": visual_emotion,
                "fused_mood": fused_mood,
            }
        ), 503

    supportive_response = llm_message

    fused_session_id = _save_fused_session(
        text_emotion=text_emotion,
        visual_emotion=visual_emotion,
        fused_mood=fused_mood,
        fused_confidence=fused_confidence,
        support_source=llm_source,
        support_message=supportive_response,
    )

    return jsonify(
        {
            "text_emotion": text_emotion,
            "text_emotion_normalized": text_normalized,
            "text_confidence": text_confidence,
            "visual_dominant_emotion": visual_emotion,
            "visual_average_confidence": visual_confidence,
            "visual_emotion_changes": emotion_changes,
            "visual_stability_score": round(
                _safe_float(visual_result.get("stability_score", 0.0), 0.0),
                4,
            ),
            "fused_mood": fused_mood,
            "fused_confidence": fused_confidence,
            "what_you_may_be_feeling": feeling_explanation,
            "supportive_response": supportive_response,
            "recommended_actions": actions,
            "distribution": visual_result.get("distribution", {}),
            "llm_source": llm_source,
            "text_session_id": text_session_id,
            "visual_session_id": visual_session_id,
            "fused_session_id": fused_session_id,
        }
    ), 200


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
