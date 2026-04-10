from config import app, client
from fastapi import UploadFile, File, Form, HTTPException
import uvicorn

import os
import tempfile

from helpers import compute_final_score


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
async def score_pronunciation(
    audio: UploadFile = File(...),
    target: str = Form(...),
):
    if not audio.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm")):
        raise HTTPException(status_code=400, detail="unsupported audio format")

    suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        raw_path = tmp.name

    try:
        # ── Directly transcribe raw audio ───────────────────────────────
        with open(raw_path, "rb") as f:
            result = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v2",
                language_code="kaz",
                tag_audio_events=False,
                timestamps_granularity="word",
            )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Processing error: {e}")

    finally:
        os.unlink(raw_path)

    words = [w for w in (result.words or []) if w.type == "word"]
    heard = result.text.strip().lower()
    heard = ''.join(c for c in heard if c.isalnum())

    if not heard:
        return {
            "score":      0,
            "heard":      "",
            "target":     target,
            "phonetic":   0,
            "confidence": 0,
            "logprob":    -999,
            "dtw_score":  None,
            "pass":       False,
            "reason":     "no speech detected",
        }

    return compute_final_score(heard, target, words)