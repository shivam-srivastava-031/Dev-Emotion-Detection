"""
Emotion detection model wrapper.
Uses a HuggingFace distilroberta model fine-tuned for emotion classification.
Detected Emotions: anger, disgust, fear, joy, neutral, sadness, surprise
"""

import re
from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Colour palette for each emotion (used by the frontend)
EMOTION_COLORS = {
    "anger": "#EF4444",
    "disgust": "#A855F7",
    "fear": "#6366F1",
    "joy": "#FACC15",
    "neutral": "#94A3B8",
    "sadness": "#3B82F6",
    "surprise": "#F97316",
}


@lru_cache(maxsize=1)
def get_classifier():
    """Load the model once and cache it."""
    print(f"[emotion_model] Loading model '{MODEL_NAME}' …")
    clf = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,          # return all labels with scores
        truncation=True,
    )
    print("[emotion_model] Model loaded ✓")
    return clf


def preprocess(text: str) -> str:
    """Basic text cleaning."""
    text = text.strip()
    # collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text)
    return text


def detect_emotion(raw_text: str) -> dict:
    """
    Run the emotion classifier on cleaned text.

    Returns:
        {
            "text": <cleaned text>,
            "emotion": <top emotion label>,
            "confidence": <float 0-1>,
            "all_scores": { label: score, ... }
        }
    """
    text = preprocess(raw_text)
    if not text:
        return {
            "text": "",
            "emotion": "neutral",
            "confidence": 1.0,
            "all_scores": {"neutral": 1.0},
        }

    clf = get_classifier()
    results = clf(text)[0]  # list of {label, score}

    all_scores = {r["label"]: round(r["score"], 4) for r in results}
    top = max(results, key=lambda r: r["score"])

    return {
        "text": text,
        "emotion": top["label"],
        "confidence": round(top["score"], 4),
        "all_scores": all_scores,
    }
