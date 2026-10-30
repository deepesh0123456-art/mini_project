import streamlit as st
from transformers import pipeline
from PIL import Image

MODEL_NAME = "Organika/sdxl-detector"


@st.cache_resource
def load_image_model():
    """
    Load the pretrained image classification model.
    Cached so it is not downloaded/loaded on every rerun.
    """
    detector = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )
    return detector


def detect_image(image: Image.Image):
    """
    Analyze an image and return class probabilities.
    """
    image = image.convert("RGB")

    detector = load_image_model()

    results = detector(image)

    scores = {
        item["label"].lower(): float(item["score"])
        for item in results
    }

    ai_score = scores.get("artificial", 0.0)
    human_score = scores.get("human", 0.0)

    prediction = (
        "Likely AI-generated"
        if ai_score >= human_score
        else "Likely human-created"
    )

    return {
        "prediction": prediction,
        "ai_probability": ai_score,
        "human_probability": human_score
    }