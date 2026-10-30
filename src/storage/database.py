"""
src/storage/database.py — SQLAlchemy models and helper functions for SQLite persistence.

Defines the ORM schema for sessions, attendees, engagement records, and
teaching suggestions. Provides convenience functions used by route handlers.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)

# Shared SQLAlchemy instance (initialised in app factory via db.init_app)
db = SQLAlchemy()


# ── ORM Models ────────────────────────────────────────────────────────────────


class Session(db.Model):  # type: ignore[name-defined]
    """Represents a single classroom session."""

    __tablename__ = "sessions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False, default="Unnamed Session")
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="active")  # active | completed | cancelled

    # Relationships
    attendees = db.relationship("Attendee", backref="session", lazy="dynamic", cascade="all, delete-orphan")
    engagement_records = db.relationship("EngagementRecord", backref="session", lazy="dynamic", cascade="all, delete-orphan")
    suggestions = db.relationship("TeachingSuggestion", backref="session", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """Serialise session to a JSON-safe dict."""
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
        }

    def __repr__(self) -> str:
        return f"<Session id={self.id} name={self.name!r} status={self.status}>"


class Attendee(db.Model):  # type: ignore[name-defined]
    """Represents a detected individual's attendance record for a session."""

    __tablename__ = "attendees"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False, index=True)
    face_id = db.Column(db.String(50), nullable=False)  # Tracker-assigned ID (not biometric)
    first_seen = db.Column(db.DateTime, nullable=True)
    last_seen = db.Column(db.DateTime, nullable=True)
    total_duration = db.Column(db.Float, default=0.0)  # Seconds
    engagement_avg = db.Column(db.Float, default=0.0)  # Average engagement score

    def to_dict(self) -> dict[str, Any]:
        """Serialise attendee to a JSON-safe dict."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "face_id": self.face_id,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "total_duration": round(self.total_duration, 1),
            "engagement_avg": round(self.engagement_avg, 1),
        }

    def __repr__(self) -> str:
        return f"<Attendee id={self.id} face_id={self.face_id!r} session={self.session_id}>"


class EngagementRecord(db.Model):  # type: ignore[name-defined]
    """A single multimodal engagement measurement at a point in time."""

    __tablename__ = "engagement_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    visual_score = db.Column(db.Float, default=0.0)
    audio_level = db.Column(db.Float, default=0.0)
    fused_score = db.Column(db.Float, default=0.0)
    level = db.Column(db.String(10), default="medium")  # high | medium | low

    def to_dict(self) -> dict[str, Any]:
        """Serialise record to a JSON-safe dict."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "visual_score": self.visual_score,
            "audio_level": self.audio_level,
            "fused_score": self.fused_score,
            "level": self.level,
        }

    def __repr__(self) -> str:
        return f"<EngagementRecord id={self.id} fused={self.fused_score:.1f} level={self.level}>"


class TeachingSuggestion(db.Model):  # type: ignore[name-defined]
    """An AI-generated teaching suggestion associated with a session."""

    __tablename__ = "teaching_suggestions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False, index=True)
    suggestion_text = db.Column(db.Text, nullable=False)
    context = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_applied = db.Column(db.Boolean, default=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise suggestion to a JSON-safe dict."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "suggestion_text": self.suggestion_text,
            "context": self.context,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_applied": self.is_applied,
        }

    def __repr__(self) -> str:
        return f"<TeachingSuggestion id={self.id} session={self.session_id}>"


# ── Helper Functions ──────────────────────────────────────────────────────────


def init_db() -> None:
    """
    Create all tables if they do not already exist.

    Should be called once inside an active application context.
    """
    db.create_all()
    logger.info("Database tables created/verified.")


def get_session(session_id: int) -> Session | None:
    """
    Fetch a session by its primary key.

    Args:
        session_id: Integer primary key of the session.

    Returns:
        Session ORM object or None if not found.
    """
    return db.session.get(Session, session_id)


def save_attendance(
    session_id: int,
    face_id: str,
    first_seen: datetime,
    last_seen: datetime,
    total_duration: float,
    engagement_avg: float,
) -> Attendee:
    """
    Create or update an Attendee record.

    If a record with the given ``session_id`` and ``face_id`` already
    exists, updates its fields instead of creating a duplicate.

    Args:
        session_id: Associated session ID.
        face_id: Tracker-assigned face identifier.
        first_seen: First detection timestamp.
        last_seen: Most recent detection timestamp.
        total_duration: Total presence duration in seconds.
        engagement_avg: Average engagement score.

    Returns:
        Created or updated Attendee instance.
    """
    existing = Attendee.query.filter_by(session_id=session_id, face_id=face_id).first()
    if existing:
        existing.last_seen = last_seen
        existing.total_duration = total_duration
        existing.engagement_avg = engagement_avg
        db.session.commit()
        return existing

    attendee = Attendee(
        session_id=session_id,
        face_id=face_id,
        first_seen=first_seen,
        last_seen=last_seen,
        total_duration=total_duration,
        engagement_avg=engagement_avg,
    )
    db.session.add(attendee)
    db.session.commit()
    return attendee


def save_engagement(
    session_id: int,
    visual_score: float,
    audio_level: float,
    fused_score: float,
    level: str,
) -> EngagementRecord:
    """
    Persist a single engagement measurement.

    Args:
        session_id: Associated session ID.
        visual_score: Visual engagement score (0–100).
        audio_level: Audio interaction level (0–1).
        fused_score: Fused engagement score (0–100).
        level: Categorical level ("high" | "medium" | "low").

    Returns:
        Created EngagementRecord instance.
    """
    record = EngagementRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        visual_score=visual_score,
        audio_level=audio_level,
        fused_score=fused_score,
        level=level,
    )
    db.session.add(record)
    db.session.commit()
    return record


def save_suggestion(
    session_id: int,
    suggestion_text: str,
    context: str = "auto-generated",
) -> TeachingSuggestion:
    """
    Persist an AI-generated teaching suggestion.

    Args:
        session_id: Associated session ID.
        suggestion_text: The suggestion content.
        context: Brief label describing how the suggestion was generated.

    Returns:
        Created TeachingSuggestion instance.
    """
    suggestion = TeachingSuggestion(
        session_id=session_id,
        suggestion_text=suggestion_text,
        context=context,
    )
    db.session.add(suggestion)
    db.session.commit()
    return suggestion


def get_session_stats(session_id: int) -> dict[str, Any]:
    """
    Compute aggregate statistics for a session.

    Args:
        session_id: Session to query.

    Returns:
        Dict with total_attendees, avg_engagement, total_records, and level counts.
    """
    records = EngagementRecord.query.filter_by(session_id=session_id).all()
    attendees = Attendee.query.filter_by(session_id=session_id).all()

    if records:
        fused_scores = [r.fused_score for r in records]
        avg_engagement = sum(fused_scores) / len(fused_scores)
        level_counts = {"high": 0, "medium": 0, "low": 0}
        for r in records:
            level_counts[r.level] = level_counts.get(r.level, 0) + 1
    else:
        avg_engagement = 0.0
        level_counts = {"high": 0, "medium": 0, "low": 0}

    return {
        "session_id": session_id,
        "total_attendees": len(attendees),
        "avg_engagement": round(avg_engagement, 1),
        "total_records": len(records),
        "level_counts": level_counts,
    }
