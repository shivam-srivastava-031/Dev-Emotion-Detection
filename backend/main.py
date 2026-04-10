"""
FastAPI application – Emotion Detection Pipeline.
Endpoints:
    POST /api/analyze           – Analyze text and store result
    GET  /api/timeline          – Retrieve emotion timeline
    GET  /api/insights          – Get generated insights
    GET  /api/emotions          – List supported emotions + colours

    POST /api/datasets/load     – Load a dataset (goemotions / meld)
    GET  /api/datasets/stats    – Dataset statistics
    GET  /api/datasets/explore  – Browse dataset records with filters

    GET  /api/behavior          – Full behavior engine data
    GET  /api/behavior/transitions – Markov transition matrix
    GET  /api/behavior/trends   – Moving average trend data
"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from database import DatasetRecord, EmotionRecord, get_db, init_db
from emotion_model import EMOTION_COLORS, detect_emotion
from insights import generate_insights
from behavior_engine import (
    compute_transition_matrix,
    compute_trend,
    get_all_behavior_data,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    detect_emotion("warmup")
    yield


app = FastAPI(title="Emotion Detection Pipeline", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class EmotionOut(BaseModel):
    id: int
    text: str
    emotion: str
    confidence: float
    created_at: str
    all_scores: dict[str, float] | None = None


class DatasetLoadRequest(BaseModel):
    source: str = Field(..., pattern="^(goemotions|meld)$")
    max_rows: int = Field(default=2000, ge=10, le=60000)


# ── Global loading state ──────────────────────────────────────────────

_loading_state: dict = {"status": "idle", "source": None, "processed": 0, "total": 0, "error": None}
_loading_lock = threading.Lock()


# ── Original Routes ───────────────────────────────────────────────────

@app.post("/api/analyze", response_model=EmotionOut)
def analyze_text(payload: AnalyzeRequest, db: Session = Depends(get_db)):
    """Analyze a piece of text and store the result."""
    result = detect_emotion(payload.text)

    record = EmotionRecord(
        text=result["text"],
        emotion=result["emotion"],
        confidence=result["confidence"],
        source="user",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return EmotionOut(
        id=record.id,
        text=record.text,
        emotion=record.emotion,
        confidence=record.confidence,
        created_at=record.created_at.isoformat(),
        all_scores=result["all_scores"],
    )


@app.get("/api/timeline", response_model=list[EmotionOut])
def get_timeline(limit: int = 50, db: Session = Depends(get_db)):
    """Return the most recent user emotion records (newest first)."""
    records = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.source == "user")
        .order_by(desc(EmotionRecord.created_at))
        .limit(limit)
        .all()
    )
    return [
        EmotionOut(
            id=r.id,
            text=r.text,
            emotion=r.emotion,
            confidence=r.confidence,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]


@app.get("/api/insights")
def get_insights(db: Session = Depends(get_db)):
    return generate_insights(db)


@app.get("/api/emotions")
def get_emotions():
    return EMOTION_COLORS


# ── Behavior Engine Routes ────────────────────────────────────────────

@app.get("/api/behavior")
def get_behavior(db: Session = Depends(get_db)):
    """Full behavior engine data: transitions, trends, spikes, patterns, insights."""
    return get_all_behavior_data(db)


@app.get("/api/behavior/transitions")
def get_transitions(db: Session = Depends(get_db)):
    """Markov transition matrix only."""
    records = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.source == "user")
        .order_by(desc(EmotionRecord.created_at))
        .limit(500)
        .all()
    )
    return compute_transition_matrix(records)


@app.get("/api/behavior/trends")
def get_trends(db: Session = Depends(get_db)):
    """Moving average trend data only."""
    records = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.source == "user")
        .order_by(desc(EmotionRecord.created_at))
        .limit(500)
        .all()
    )
    return compute_trend(records)


# ── Dataset Routes ────────────────────────────────────────────────────

@app.post("/api/datasets/load")
def load_dataset_endpoint(payload: DatasetLoadRequest, db: Session = Depends(get_db)):
    """
    Trigger dataset download and batch analysis.
    Runs in a background thread; poll /api/datasets/load/status for progress.
    """
    global _loading_state

    with _loading_lock:
        if _loading_state["status"] == "loading":
            raise HTTPException(400, "A dataset is already being loaded.")

        _loading_state = {
            "status": "loading",
            "source": payload.source,
            "processed": 0,
            "total": payload.max_rows,
            "error": None,
        }

    def _bg_load():
        global _loading_state
        try:
            from dataset_loader import load_goemotions, load_meld
            from database import SessionLocal, DatasetRecord

            loader = load_goemotions if payload.source == "goemotions" else load_meld
            session = SessionLocal()
            batch = []
            count = 0

            for record_data in loader(max_rows=payload.max_rows):
                # Run through BERT
                result = detect_emotion(record_data["text"])

                batch.append(DatasetRecord(
                    text=record_data["text"],
                    source=record_data["source"],
                    ground_truth=record_data["ground_truth"],
                    mapped_emotion=record_data["mapped_emotion"],
                    predicted_emotion=result["emotion"],
                    confidence=result["confidence"],
                    speaker=record_data.get("speaker"),
                    conversation_id=record_data.get("conversation_id"),
                ))

                count += 1
                _loading_state["processed"] = count

                # Batch insert every 50 records
                if len(batch) >= 50:
                    session.bulk_save_objects(batch)
                    session.commit()
                    batch = []

            # Flush remaining
            if batch:
                session.bulk_save_objects(batch)
                session.commit()

            session.close()
            _loading_state["status"] = "done"
            _loading_state["processed"] = count
            log.info(f"Dataset '{payload.source}' loaded: {count} records")

        except Exception as e:
            log.error(f"Dataset load error: {e}")
            _loading_state["status"] = "error"
            _loading_state["error"] = str(e)

    thread = threading.Thread(target=_bg_load, daemon=True)
    thread.start()

    return {"message": f"Loading {payload.source}…", "status": "loading"}


@app.get("/api/datasets/load/status")
def dataset_load_status():
    """Poll the status of a dataset loading job."""
    return _loading_state


@app.get("/api/datasets/stats")
def dataset_stats(db: Session = Depends(get_db)):
    """Per-source dataset statistics: counts, accuracy vs ground truth, distributions."""
    sources = db.query(DatasetRecord.source, func.count(DatasetRecord.id)).group_by(DatasetRecord.source).all()

    stats = {}
    for source, total in sources:
        # Accuracy: predicted_emotion matches mapped_emotion
        correct = (
            db.query(func.count(DatasetRecord.id))
            .filter(
                DatasetRecord.source == source,
                DatasetRecord.predicted_emotion == DatasetRecord.mapped_emotion,
            )
            .scalar()
        )

        # Emotion distribution (predicted)
        dist_rows = (
            db.query(DatasetRecord.predicted_emotion, func.count(DatasetRecord.id))
            .filter(DatasetRecord.source == source)
            .group_by(DatasetRecord.predicted_emotion)
            .all()
        )
        distribution = {e: c for e, c in dist_rows}

        # Ground truth distribution
        gt_rows = (
            db.query(DatasetRecord.mapped_emotion, func.count(DatasetRecord.id))
            .filter(DatasetRecord.source == source)
            .group_by(DatasetRecord.mapped_emotion)
            .all()
        )
        gt_distribution = {e: c for e, c in gt_rows}

        stats[source] = {
            "total": total,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "predicted_distribution": distribution,
            "ground_truth_distribution": gt_distribution,
        }

    return stats


@app.get("/api/datasets/explore")
def explore_dataset(
    source: str | None = Query(None),
    emotion: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    db: Session = Depends(get_db),
):
    """Browse dataset records with filters."""
    query = db.query(DatasetRecord)
    if source:
        query = query.filter(DatasetRecord.source == source)
    if emotion:
        query = query.filter(DatasetRecord.predicted_emotion == emotion)

    total = query.count()
    records = (
        query
        .order_by(desc(DatasetRecord.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
        "records": [
            {
                "id": r.id,
                "text": r.text,
                "source": r.source,
                "ground_truth": r.ground_truth,
                "mapped_emotion": r.mapped_emotion,
                "predicted_emotion": r.predicted_emotion,
                "confidence": r.confidence,
                "speaker": r.speaker,
                "conversation_id": r.conversation_id,
            }
            for r in records
        ],
    }
