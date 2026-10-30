import streamlit as st
import librosa
import numpy as np
import io
from transformers import pipeline

MODEL_NAME = "Hemgg/Deepfake-audio-detection"


@st.cache_resource
def load_audio_model():
    return pipeline(
        "audio-classification",
        model=MODEL_NAME,
        device=-1
    )


def detect_audio(audio_bytes):
    # Load audio and resample to 16 kHz mono
    audio, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio is empty.")

    # Limit processing time for the one-day prototype
    max_samples = 15 * 16000
    audio = audio[:max_samples]

    detector = load_audio_model()

    results = detector({
        "raw": audio.astype(np.float32),
        "sampling_rate": 16000
    })

    # Identify the class labels returned by the model
    ai_score = None
    real_score = None

    for item in results:
        label = item["label"].lower()
        score = float(item["score"])

        if any(word in label for word in ["fake", "spoof", "artificial", "synthetic","generated", "aivoice"]):
            ai_score = score
        elif any(word in label for word in ["real", "human", "bonafide", "bona fide","genuine", "humanvoice"]):
            real_score = score

    return {
        "results": results,
        "ai_probability": ai_score,
        "human_probability": real_score
    }