# Integrated API (Text + Visual + LLM)

Run from project root:

```powershell
pip install -r api/requirements.txt
python api/app.py
```

Open the tester UI:

- `http://127.0.0.1:8000/`

## LLM Provider (OpenRouter)

Set these in project root `.env`:

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openai/gpt-4o-mini
USE_OPENROUTER=true
AI_MODE=true
```

Optional headers for OpenRouter rankings/analytics:

```env
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=MoodSense
```

Optional fallback provider:

```env
USE_GEMINI=false
GEMINI_API_KEY=
```

## Endpoints

- `GET /health`
- `POST /predict/text`
  - body: `{ "text": "...", "save_session": true }`
- `POST /visual/start`
- `POST /visual/stop`
- `POST /fuse`
  - optional body:
    - `{ "text_result": {"mood": "sad", "confidence": 0.8}, "visual_result": {"mood": "neutral", "confidence": 0.6} }`

`/fuse` combines latest saved text/visual sessions, computes trend, then generates LLM guidance.
