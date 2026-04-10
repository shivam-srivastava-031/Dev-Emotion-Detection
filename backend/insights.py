"""
Pattern Detection Engine + Insight Generator.
Analyses stored emotion records and produces human-readable insights.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc
from sqlalchemy.orm import Session

from database import EmotionRecord


# ── helpers ────────────────────────────────────────────────────────────

def _records_since(db: Session, hours: int = 24):
    """Fetch records created within the last `hours` hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return (
        db.query(EmotionRecord)
        .filter(EmotionRecord.created_at >= cutoff)
        .order_by(desc(EmotionRecord.created_at))
        .all()
    )


def _all_records(db: Session, limit: int = 200):
    return (
        db.query(EmotionRecord)
        .order_by(desc(EmotionRecord.created_at))
        .limit(limit)
        .all()
    )


# ── pattern detectors ────────────────────────────────────────────────

def dominant_emotion(records) -> dict | None:
    """Find the most frequently occurring emotion."""
    if not records:
        return None
    counts = Counter(r.emotion for r in records)
    top_emotion, top_count = counts.most_common(1)[0]
    total = sum(counts.values())
    return {
        "type": "dominant_emotion",
        "title": "Dominant Emotion",
        "emotion": top_emotion,
        "percentage": round(top_count / total * 100, 1),
        "description": (
            f"Your dominant emotion is **{top_emotion}** "
            f"({top_count}/{total} entries, {round(top_count / total * 100, 1)}%)."
        ),
    }


def emotion_streak(records) -> dict | None:
    """Detect consecutive entries with the same emotion."""
    if len(records) < 2:
        return None
    streak_emotion = records[0].emotion
    streak_len = 1
    for r in records[1:]:
        if r.emotion == streak_emotion:
            streak_len += 1
        else:
            break
    if streak_len < 2:
        return None
    return {
        "type": "emotion_streak",
        "title": "Emotion Streak",
        "emotion": streak_emotion,
        "streak_length": streak_len,
        "description": (
            f"You've been feeling **{streak_emotion}** for "
            f"the last **{streak_len}** consecutive entries."
        ),
    }


def emotional_volatility(records) -> dict | None:
    """Measure how many unique emotions appeared recently."""
    if len(records) < 3:
        return None
    unique = set(r.emotion for r in records)
    ratio = round(len(unique) / len(records), 2)
    if ratio > 0.6:
        level = "high"
        desc_text = "Your emotional state has been **highly varied** recently. Take some time to ground yourself. 🧘"
    elif ratio > 0.35:
        level = "moderate"
        desc_text = "You're experiencing a **healthy mix** of emotions."
    else:
        level = "low"
        desc_text = "Your emotions have been quite **stable** recently."
    return {
        "type": "emotional_volatility",
        "title": "Emotional Volatility",
        "level": level,
        "unique_emotions": len(unique),
        "total_entries": len(records),
        "description": desc_text,
    }


def emotion_distribution(records) -> dict | None:
    """Full percentage breakdown of emotions."""
    if not records:
        return None
    counts = Counter(r.emotion for r in records)
    total = sum(counts.values())
    dist = {k: round(v / total * 100, 1) for k, v in counts.most_common()}
    return {
        "type": "emotion_distribution",
        "title": "Emotion Distribution",
        "distribution": dist,
        "description": "Percentage breakdown of all recorded emotions.",
    }


def recent_shift(records) -> dict | None:
    """Detect if the most recent emotion differs from the previous trend."""
    if len(records) < 4:
        return None
    latest = records[0].emotion
    previous_counts = Counter(r.emotion for r in records[1:6])
    prev_dominant, _ = previous_counts.most_common(1)[0]
    if latest != prev_dominant:
        return {
            "type": "recent_shift",
            "title": "Mood Shift Detected",
            "from_emotion": prev_dominant,
            "to_emotion": latest,
            "description": (
                f"Your latest entry shows **{latest}**, shifting from "
                f"a recent trend of **{prev_dominant}**."
            ),
        }
    return None


# ── public API ────────────────────────────────────────────────────────

ALL_DETECTORS = [
    dominant_emotion,
    emotion_streak,
    emotional_volatility,
    emotion_distribution,
    recent_shift,
]


def generate_insights(db: Session) -> list[dict]:
    """Run all pattern detectors and return non‑None results."""
    records = _all_records(db)
    insights: list[dict] = []
    for detector in ALL_DETECTORS:
        result = detector(records)
        if result is not None:
            insights.append(result)
    return insights
