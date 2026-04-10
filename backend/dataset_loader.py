"""
Dataset Loader — GoEmotions + MELD (EmotionLines).
Downloads via HuggingFace `datasets`, normalizes, maps labels.
"""

import logging
from typing import Generator

log = logging.getLogger(__name__)

# ── GoEmotions 28-class → 7-class mapping ────────────────────────────
# Maps every GoEmotions fine-grained label to one of our 7 emotions.
GOEMOTIONS_TO_7CLASS: dict[str, str] = {
    # joy-family
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "caring": "joy",
    "desire": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    # sadness-family
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    # anger-family
    "anger": "anger",
    "annoyance": "anger",
    # fear-family
    "fear": "fear",
    "nervousness": "fear",
    # surprise-family
    "surprise": "surprise",
    "realization": "surprise",
    "curiosity": "surprise",
    "confusion": "surprise",
    # disgust-family
    "disgust": "disgust",
    "disapproval": "disgust",
    "embarrassment": "disgust",
    # neutral
    "neutral": "neutral",
}

# MELD emotion labels are already close to our 7-class scheme
MELD_LABEL_MAP: dict[str, str] = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}


def _goemotions_label_names() -> list[str]:
    """Return the GoEmotions simplified label names in order."""
    return [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise",
        "neutral",
    ]


def load_goemotions(max_rows: int | None = 2000) -> Generator[dict, None, None]:
    """
    Yield normalized records from GoEmotions (simplified).
    Each record: {text, source, ground_truth, mapped_emotion}

    Args:
        max_rows: Cap the number of rows (None = all ~58K).
                  Default 2000 for fast demo loading.
    """
    from datasets import load_dataset

    log.info("Downloading GoEmotions (simplified) …")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    label_names = _goemotions_label_names()

    count = 0
    for row in ds:
        if max_rows and count >= max_rows:
            break
        text = row["text"].strip()
        if not text:
            continue

        # GoEmotions simplified stores labels as list of int indices
        label_ids = row["labels"]
        if not label_ids:
            continue

        # Take first (primary) label
        primary_label = label_names[label_ids[0]] if label_ids[0] < len(label_names) else "neutral"
        mapped = GOEMOTIONS_TO_7CLASS.get(primary_label, "neutral")

        yield {
            "text": text,
            "source": "goemotions",
            "ground_truth": primary_label,
            "mapped_emotion": mapped,
            "speaker": None,
            "conversation_id": None,
        }
        count += 1

    log.info(f"GoEmotions: yielded {count} records")


def load_meld(max_rows: int | None = 1000) -> Generator[dict, None, None]:
    """
    Yield normalized records from MELD (EmotionLines).
    Each record: {text, source, ground_truth, mapped_emotion, speaker, conversation_id}
    Reads directly from MELD GitHub CSV using pandas to avoid downloading 11GB raw dataset.

    Args:
        max_rows: Cap the number of rows (None = all ~10K).
                  Default 1000 for fast demo loading.
    """
    import pandas as pd

    log.info("Downloading MELD CSV …")
    url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        log.error(f"Failed to load MELD from CSV: {e}")
        return

    count = 0
    for _, row in df.iterrows():
        if max_rows and count >= max_rows:
            break

        text = str(row.get("Utterance", "")).strip()
        if not text or text == "nan":
            continue

        emotion_raw = str(row.get("Emotion", "neutral")).strip().lower()
        mapped = MELD_LABEL_MAP.get(emotion_raw, "neutral")
        speaker = str(row.get("Speaker", ""))
        dialogue_id = str(row.get("Dialogue_ID", ""))

        yield {
            "text": text,
            "source": "meld",
            "ground_truth": emotion_raw,
            "mapped_emotion": mapped,
            "speaker": speaker if speaker and speaker != "nan" else None,
            "conversation_id": dialogue_id if dialogue_id and dialogue_id != "nan" else None,
        }
        count += 1

    log.info(f"MELD: yielded {count} records")
