import numpy as np 
from phenome_distances import PHONEME_DISTANCES, KAZAKH_WORDS
import librosa
import tempfile
import soundfile as sf

import math

def phoneme_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return PHONEME_DISTANCES.get((a, b), 1.0)


def phonetic_edit_distance(heard: str, target: str) -> float:
    h    = heard.strip().lower()
    t    = target.strip().lower()
    rows = len(h) + 1
    cols = len(t) + 1
    dp   = np.zeros((rows, cols))
    dp[0, :] = range(cols)
    dp[:, 0] = range(rows)
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i, j] = min(
                dp[i-1, j]   + 1.0,
                dp[i, j-1]   + 1.0,
                dp[i-1, j-1] + phoneme_cost(h[i-1], t[j-1]),
            )
    return dp[rows-1, cols-1]


def phonetic_score(heard: str, target: str) -> int:
    if not heard:
        return 0
    dist    = phonetic_edit_distance(heard, target)
    max_len = max(len(heard), len(target))
    return max(0, min(100, round(100 * (1 - dist / max_len))))

# def preprocess_audio(input_path: str) -> str:
#     """
#     Load audio with librosa, normalize amplitude, trim leading/trailing
#     silence, and write the result to a new temp WAV file.
#     Returns the path to the cleaned file.
#     """
#     y, sr = librosa.load(input_path, sr=16_000, mono=True)

#     # Amplitude normalisation — peak-normalize to [-1, 1]
#     peak = np.max(np.abs(y))
#     if peak > 0:
#         y = y / peak

#     # Trim silence (top_db=20 is a good default; lower = more aggressive)
#     y_trimmed, _ = librosa.effects.trim(y, top_db=20)

#     # Write cleaned audio to a new temp WAV
#     cleaned = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     sf.write(cleaned.name, y_trimmed, sr)
#     return cleaned.name, y_trimmed, sr

def compute_final_score(
    heard: str,
    target: str,
    words: list,
) -> dict:
    # Signal 1 — acoustic confidence from logprobs
    if words:
        avg_logprob      = sum(w.logprob for w in words) / len(words)
        confidence_score = round(math.exp(avg_logprob) * 100)
    else:
        avg_logprob      = -999
        confidence_score = 0

    # Signal 2 — phonetic similarity of transcript vs target
    kazakh_target = KAZAKH_WORDS[target]
    phon = phonetic_score(heard, kazakh_target)

    return {
        "score":       phon,
        "heard":       heard,
        "target":      target,
        "phonetic":    phon,
        "confidence":  confidence_score,
        "logprob":     round(avg_logprob, 3),         # None if no reference exists
        "pass":        phon >= 60,
    }