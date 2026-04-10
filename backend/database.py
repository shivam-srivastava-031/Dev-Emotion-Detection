"""
Database models and session management for the Emotion Pipeline.
Uses SQLAlchemy with SQLite.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DB_PATH = os.path.join(os.path.dirname(__file__), "emotion_data.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class EmotionRecord(Base):
    """A single analyzed text entry with its detected emotion."""

    __tablename__ = "emotion_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    emotion = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    source = Column(String(50), default="user", nullable=False)  # "user" | "goemotions" | "meld"
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )


class DatasetRecord(Base):
    """A record from an external dataset with ground truth + prediction."""

    __tablename__ = "dataset_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    source = Column(String(50), nullable=False)           # "goemotions" | "meld"
    ground_truth = Column(String(50), nullable=True)       # original label from dataset
    mapped_emotion = Column(String(50), nullable=True)     # mapped to 7-class
    predicted_emotion = Column(String(50), nullable=False)  # our BERT prediction
    confidence = Column(Float, nullable=False)
    speaker = Column(String(100), nullable=True)           # for MELD conversations
    conversation_id = Column(String(100), nullable=True)   # for MELD conversations
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
