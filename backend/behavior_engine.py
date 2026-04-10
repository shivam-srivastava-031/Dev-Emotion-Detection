"""
Behavior Pattern Engine — the USP layer.

Provides:
  1. Markov chain emotion transitions (detect loops)
  2. Spike detection (sudden emotion bursts)
  3. Moving average trend (emotional drift / instability)
  4. Time-of-day clustering ("late night sadness", etc.)
  5. Context window analysis (last N messages)
  6. Rich human-nature insight generator
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import desc
from sqlalchemy.orm import Session

from database import EmotionRecord

log = logging.getLogger(__name__)

# Canonical emotion list
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# ─────────────────────────────────────────────────────────────────────
#  1. MARKOV CHAIN — Emotion Transition Matrix
# ─────────────────────────────────────────────────────────────────────

def compute_transition_matrix(records: list) -> dict:
    """
    Build a Markov transition matrix from sequential emotion records.
    Returns:
      {
        "matrix": [[float, ...], ...],   # 7×7 transition probabilities
        "labels": ["anger", ...],
        "transitions": [{"from": "joy", "to": "sadness", "probability": 0.32}, ...],
        "loops": [{"pattern": ["sad","angry","neutral","sad"], "count": N}, ...]
      }
    """
    n = len(EMOTIONS)
    counts = np.zeros((n, n), dtype=float)

    # Records arrive newest-first; reverse to get chronological order
    chronological = list(reversed(records))

    for i in range(len(chronological) - 1):
        curr = chronological[i].emotion
        nxt = chronological[i + 1].emotion
        if curr in EMOTION_IDX and nxt in EMOTION_IDX:
            counts[EMOTION_IDX[curr]][EMOTION_IDX[nxt]] += 1

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    matrix = (counts / row_sums).tolist()

    # Extract top transitions
    transitions = []
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0.05:
                transitions.append({
                    "from": EMOTIONS[i],
                    "to": EMOTIONS[j],
                    "probability": round(matrix[i][j], 3),
                })
    transitions.sort(key=lambda t: t["probability"], reverse=True)

    # Detect repeating loops (length 3-5)
    loops = _detect_loops(chronological)

    return {
        "matrix": [[round(v, 3) for v in row] for row in matrix],
        "labels": EMOTIONS,
        "transitions": transitions[:20],
        "loops": loops,
    }


def _detect_loops(chronological: list, min_len: int = 3, max_len: int = 5) -> list[dict]:
    """Find repeating emotion sub-sequences (loops)."""
    emotions_seq = [r.emotion for r in chronological]
    loop_counts: Counter = Counter()

    for length in range(min_len, max_len + 1):
        for i in range(len(emotions_seq) - length * 2 + 1):
            pattern = tuple(emotions_seq[i : i + length])
            # Check if pattern repeats immediately after
            next_seg = tuple(emotions_seq[i + length : i + length * 2])
            if pattern == next_seg:
                loop_counts[pattern] += 1

    loops = [
        {"pattern": list(p), "count": c}
        for p, c in loop_counts.most_common(5)
        if c >= 2
    ]
    return loops


# ─────────────────────────────────────────────────────────────────────
#  2. SPIKE DETECTION
# ─────────────────────────────────────────────────────────────────────

def detect_spikes(records: list, window: int = 10, threshold: float = 0.6) -> list[dict]:
    """
    Detect sudden bursts where one emotion dominates a short window.
    A spike = one emotion appears ≥ threshold% of a sliding window.

    Returns list of {emotion, window_start, window_end, frequency, description}
    """
    if len(records) < window:
        return []

    chronological = list(reversed(records))
    spikes = []
    seen_spikes = set()

    for i in range(len(chronological) - window + 1):
        window_slice = chronological[i : i + window]
        counts = Counter(r.emotion for r in window_slice)
        top_emotion, top_count = counts.most_common(1)[0]
        freq = top_count / window

        if freq >= threshold and top_emotion != "neutral":
            key = (top_emotion, i // window)
            if key not in seen_spikes:
                seen_spikes.add(key)
                spikes.append({
                    "emotion": top_emotion,
                    "frequency": round(freq * 100, 1),
                    "window_size": window,
                    "position": i,
                    "description": (
                        f"**{top_emotion.capitalize()} burst** detected — "
                        f"{top_count}/{window} entries ({round(freq * 100)}%) "
                        f"in a short window."
                    ),
                })

    return spikes[:10]


# ─────────────────────────────────────────────────────────────────────
#  3. MOVING AVERAGE TREND (Emotional drift / instability)
# ─────────────────────────────────────────────────────────────────────

# Valence mapping: how "positive" each emotion is (0 = very negative, 1 = very positive)
EMOTION_VALENCE = {
    "anger": 0.1,
    "disgust": 0.15,
    "fear": 0.2,
    "sadness": 0.15,
    "neutral": 0.5,
    "surprise": 0.6,
    "joy": 0.9,
}


def compute_trend(records: list, window: int = 5) -> dict:
    """
    Compute a moving average of emotional valence over time.
    Returns:
      {
        "data_points": [{"index": i, "valence": 0.xx, "emotion": "...", "timestamp": "..."}, ...],
        "moving_average": [float, ...],
        "trend": "improving" | "declining" | "stable" | "unstable",
        "instability_score": float (0-1),
        "description": "..."
      }
    """
    if len(records) < 3:
        return {"data_points": [], "moving_average": [], "trend": "stable",
                "instability_score": 0, "description": "Not enough data yet."}

    chronological = list(reversed(records))
    data_points = []
    valences = []

    for i, r in enumerate(chronological):
        v = EMOTION_VALENCE.get(r.emotion, 0.5)
        valences.append(v)
        data_points.append({
            "index": i,
            "valence": v,
            "emotion": r.emotion,
            "timestamp": r.created_at.isoformat() if r.created_at else "",
        })

    # Moving average
    ma = []
    for i in range(len(valences)):
        start = max(0, i - window + 1)
        ma.append(round(float(np.mean(valences[start : i + 1])), 3))

    # Trend detection: compare first and last quarter averages
    q_len = max(1, len(ma) // 4)
    first_q = np.mean(ma[:q_len])
    last_q = np.mean(ma[-q_len:])
    diff = last_q - first_q

    # Instability = standard deviation of valence
    instability = float(np.std(valences))

    if instability > 0.3:
        trend = "unstable"
        desc_text = "Your emotions have been **highly unstable**. Consider journaling or talking to someone. 💬"
    elif diff > 0.15:
        trend = "improving"
        desc_text = "Your emotional state is **improving** — more positive emotions recently. 🌱"
    elif diff < -0.15:
        trend = "declining"
        desc_text = "Your emotional state is **declining** — more negative emotions recently. Take care. 🫂"
    else:
        trend = "stable"
        desc_text = "Your emotions have been relatively **stable**."

    return {
        "data_points": data_points,
        "moving_average": ma,
        "trend": trend,
        "instability_score": round(instability, 3),
        "description": desc_text,
    }


# ─────────────────────────────────────────────────────────────────────
#  4. TIME-OF-DAY CLUSTERING
# ─────────────────────────────────────────────────────────────────────

TIME_SLOTS = {
    "early_morning": (5, 8),
    "morning": (8, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 24),
    "late_night": (0, 5),
}


def _get_time_slot(hour: int) -> str:
    for slot, (start, end) in TIME_SLOTS.items():
        if slot == "late_night":
            if hour >= 0 and hour < 5:
                return slot
        elif start <= hour < end:
            return slot
    return "unknown"


def time_of_day_patterns(records: list) -> list[dict]:
    """
    Group emotions by time-of-day and surface patterns like
    "late night sadness" or "morning joy".

    Returns list of {time_slot, dominant_emotion, count, total, percentage, description}
    """
    slot_emotions: dict[str, list[str]] = defaultdict(list)

    for r in records:
        if r.created_at:
            hour = r.created_at.hour
            slot = _get_time_slot(hour)
            slot_emotions[slot].append(r.emotion)

    patterns = []
    for slot, emotions in slot_emotions.items():
        if len(emotions) < 2:
            continue
        counts = Counter(emotions)
        dominant, dom_count = counts.most_common(1)[0]
        pct = round(dom_count / len(emotions) * 100, 1)

        if pct >= 40 and dominant != "neutral":
            slot_label = slot.replace("_", " ").title()
            patterns.append({
                "time_slot": slot,
                "dominant_emotion": dominant,
                "count": dom_count,
                "total": len(emotions),
                "percentage": pct,
                "description": (
                    f"**{slot_label} {dominant}** — "
                    f"{dominant} appears {pct}% of the time during {slot_label.lower()} hours."
                ),
            })

    patterns.sort(key=lambda p: p["percentage"], reverse=True)
    return patterns


# ─────────────────────────────────────────────────────────────────────
#  5. CONTEXT WINDOW (Last N messages)
# ─────────────────────────────────────────────────────────────────────

def context_summary(records: list, n: int = 10) -> dict:
    """
    Summarize the emotional context of the last N messages.
    Returns:
      {
        "window_size": N,
        "emotions": [str, ...],
        "dominant": str,
        "valence_avg": float,
        "description": str
      }
    """
    recent = records[:n]  # already newest-first
    if not recent:
        return {"window_size": 0, "emotions": [], "dominant": "neutral",
                "valence_avg": 0.5, "description": "No recent entries."}

    emotions = [r.emotion for r in recent]
    counts = Counter(emotions)
    dominant, _ = counts.most_common(1)[0]
    valence_avg = round(float(np.mean([EMOTION_VALENCE.get(e, 0.5) for e in emotions])), 3)

    return {
        "window_size": len(recent),
        "emotions": emotions,
        "dominant": dominant,
        "valence_avg": valence_avg,
        "description": (
            f"In the last **{len(recent)}** messages, the dominant emotion is "
            f"**{dominant}** with an average valence of **{valence_avg:.2f}**."
        ),
    }


# ─────────────────────────────────────────────────────────────────────
#  6. RICH INSIGHT GENERATOR (Human Nature Layer)
# ─────────────────────────────────────────────────────────────────────

def generate_behavior_insights(records: list) -> list[dict]:
    """
    Combine all pattern engines into rich, human-readable insights.
    This is the rule-based + ML hybrid layer.
    """
    insights: list[dict] = []

    if len(records) < 3:
        return insights

    # ── Spike insights ──
    spikes = detect_spikes(records)
    for spike in spikes[:3]:
        insights.append({
            "type": "spike",
            "icon": "⚡",
            "title": f"{spike['emotion'].capitalize()} Burst",
            "severity": "high" if spike["frequency"] > 70 else "medium",
            "description": spike["description"],
        })

    # ── Trend insight ──
    trend = compute_trend(records)
    if trend["trend"] != "stable" or trend["instability_score"] > 0.2:
        insights.append({
            "type": "trend",
            "icon": "📈" if trend["trend"] == "improving" else "📉" if trend["trend"] == "declining" else "🌊",
            "title": f"Emotional Trend: {trend['trend'].capitalize()}",
            "severity": "low" if trend["trend"] == "improving" else "high",
            "description": trend["description"],
        })

    # ── Time-of-day insights ──
    tod_patterns = time_of_day_patterns(records)
    for pattern in tod_patterns[:3]:
        slot_label = pattern["time_slot"].replace("_", " ").title()
        if pattern["dominant_emotion"] in ("sadness", "anger", "fear"):
            insight_text = (
                f"You show repeated **{pattern['dominant_emotion']}** during "
                f"**{slot_label.lower()}** hours → possible **"
                f"{'loneliness' if pattern['dominant_emotion'] == 'sadness' else 'stress'} pattern**."
            )
        else:
            insight_text = pattern["description"]

        insights.append({
            "type": "time_pattern",
            "icon": "🌙" if "night" in pattern["time_slot"] else "☀️",
            "title": f"{slot_label} {pattern['dominant_emotion'].capitalize()}",
            "severity": "medium" if pattern["dominant_emotion"] in ("sadness", "anger", "fear") else "low",
            "description": insight_text,
        })

    # ── Loop / transition insights ──
    transitions = compute_transition_matrix(records)
    for loop in transitions["loops"][:2]:
        pattern_str = " → ".join(loop["pattern"])
        insights.append({
            "type": "loop",
            "icon": "🔄",
            "title": "Emotional Loop Detected",
            "severity": "medium",
            "description": (
                f"Repeating pattern: **{pattern_str}** "
                f"(detected {loop['count']} times). This may indicate a cycle worth exploring."
            ),
        })

    # ── Context window insight ──
    ctx = context_summary(records)
    if ctx["valence_avg"] < 0.3:
        insights.append({
            "type": "context",
            "icon": "🫂",
            "title": "Recent Emotional Low",
            "severity": "high",
            "description": (
                f"Your last {ctx['window_size']} entries show a low "
                f"emotional valence (**{ctx['valence_avg']:.2f}**). "
                f"Consider reaching out to someone or taking a break."
            ),
        })
    elif ctx["valence_avg"] > 0.75:
        insights.append({
            "type": "context",
            "icon": "✨",
            "title": "Positive Momentum",
            "severity": "low",
            "description": (
                f"Your last {ctx['window_size']} entries are strongly positive "
                f"(valence **{ctx['valence_avg']:.2f}**). Keep it up!"
            ),
        })

    return insights


# ─────────────────────────────────────────────────────────────────────
#  PUBLIC: Fetch records and run everything
# ─────────────────────────────────────────────────────────────────────

def get_all_behavior_data(db: Session, limit: int = 500) -> dict:
    """
    Run the full behavior engine and return all data for the frontend.
    """
    records = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.source == "user")
        .order_by(desc(EmotionRecord.created_at))
        .limit(limit)
        .all()
    )

    if len(records) < 3:
        return {
            "transitions": {"matrix": [], "labels": EMOTIONS, "transitions": [], "loops": []},
            "trend": {"data_points": [], "moving_average": [], "trend": "stable",
                      "instability_score": 0, "description": "Not enough data."},
            "spikes": [],
            "time_patterns": [],
            "context": {"window_size": 0, "emotions": [], "dominant": "neutral",
                        "valence_avg": 0.5, "description": "Not enough data."},
            "insights": [],
        }

    return {
        "transitions": compute_transition_matrix(records),
        "trend": compute_trend(records),
        "spikes": detect_spikes(records),
        "time_patterns": time_of_day_patterns(records),
        "context": context_summary(records),
        "insights": generate_behavior_insights(records),
    }
