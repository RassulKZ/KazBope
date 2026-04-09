from config import app, client
from fastapi import UploadFile, File, Form, HTTPException
import uvicorn

import os
import tempfile

from helpers import compute_final_score, preprocess_audio

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
async def score_pronunciation(
    audio: UploadFile = File(...),        # wav file from Flutter
    target: str       = Form(...),        # e.g. "алма"
):
    if not audio.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm")):
        raise HTTPException(status_code=400, detail="unsupported audio format")

    suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        raw_path = tmp.name

    cleaned_path = None

    try:
        # ── 1. Normalise + trim ──────────────────────────────────────────
        cleaned_path, y_clean, sr = preprocess_audio(raw_path)

        # ── 3. Transcribe the cleaned file ───────────────────────────────
        with open(cleaned_path, "rb") as f:
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
        if cleaned_path and os.path.exists(cleaned_path):
            os.unlink(cleaned_path)

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




# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)