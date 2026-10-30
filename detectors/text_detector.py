
import streamlit as st
from transformers import pipeline

MODEL_NAME = "ahmediqbal/ai-text-detector-model"


@st.cache_resource
def load_text_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
        top_k=None
    )


def detect_text(text):

    if len(text.strip()) < 20:
        raise ValueError("Please enter at least 20 characters.")

    detector = load_text_model()

    results = detector(
        text[:2000],
        truncation=True,
        max_length=512
    )[0]

    return results
