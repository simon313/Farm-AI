"""
memory/db.py
All SQLite database access for the Bardo Farm AI System.

This is the ONLY file that touches the database. No raw SQL anywhere else.
All list and dict fields on dataclasses are stored as JSON strings in SQLite.
Conversion between dataclasses and SQLite rows happens only here.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Generator, List, Optional

from memory.models import Animal, Content, Moment, Person

logger = logging.getLogger(__name__)

# Path to the migration file relative to this module's location
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


# ─── CONNECTION MANAGEMENT ─────────────────────────────────────────────────────

class Database:
    """
    Manages the SQLite connection and exposes CRUD operations for all entities.
    Instantiate once at startup and pass the instance to agents that need it.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._run_migrations()
        logger.info("Database ready at %s", db_path)

    def _run_migrations(self) -> None:
        """Apply all migration SQL files in order if tables do not yet exist."""
        migration_file = _MIGRATIONS_DIR / "001_initial.sql"
        sql = migration_file.read_text()
        with self._connect() as conn:
            conn.executescript(sql)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ─── SERIALIZATION HELPERS ─────────────────────────────────────────────────────

def _to_json(value) -> str:
    return json.dumps(value)


def _from_json(value: Optional[str], default):
    if value is None:
        return default
    return json.loads(value)


def _date_to_str(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d is not None else None


def _str_to_date(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None


def _datetime_to_str(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _str_to_datetime(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


# ─── MOMENT CRUD ───────────────────────────────────────────────────────────────

def _moment_to_row(m: Moment) -> dict:
    return {
        "id": m.id,
        "timestamp": _datetime_to_str(m.timestamp),
        "duration_seconds": m.duration_seconds,
        "camera_id": m.camera_id,
        "location_label": m.location_label,
        "clip_path": m.clip_path,
        "thumbnail_path": m.thumbnail_path,
        "audio_path": m.audio_path,
        "technical_score": m.technical_score,
        "activity_score": m.activity_score,
        "interest_score": m.interest_score,
        "farm_vibe_score": m.farm_vibe_score,
        "overall_score": m.overall_score,
        "tags": _to_json(m.tags),
        "animals_present": _to_json(m.animals_present),
        "emotional_register": _to_json(m.emotional_register),
        "narrative_note": m.narrative_note,
        "suggested_use": m.suggested_use,
        "conditions": _to_json(m.conditions),
        "passed_gates": int(m.passed_gates),
        "reviewed": int(m.reviewed),
        "used_in_content": _to_json(m.used_in_content),
        "archived": int(m.archived),
    }


def _row_to_moment(row: sqlite3.Row) -> Moment:
    return Moment(
        id=row["id"],
        timestamp=_str_to_datetime(row["timestamp"]),
        duration_seconds=row["duration_seconds"],
        camera_id=row["camera_id"],
        location_label=row["location_label"],
        clip_path=row["clip_path"],
        thumbnail_path=row["thumbnail_path"],
        audio_path=row["audio_path"],
        technical_score=row["technical_score"],
        activity_score=row["activity_score"],
        interest_score=row["interest_score"],
        farm_vibe_score=row["farm_vibe_score"],
        overall_score=row["overall_score"],
        tags=_from_json(row["tags"], []),
        animals_present=_from_json(row["animals_present"], []),
        emotional_register=_from_json(row["emotional_register"], []),
        narrative_note=row["narrative_note"],
        suggested_use=row["suggested_use"],
        conditions=_from_json(row["conditions"], {}),
        passed_gates=bool(row["passed_gates"]),
        reviewed=bool(row["reviewed"]),
        used_in_content=_from_json(row["used_in_content"], []),
        archived=bool(row["archived"]),
    )


def save_moment(db: Database, moment: Moment) -> None:
    """Insert or replace a Moment record."""
    row = _moment_to_row(moment)
    sql = """
        INSERT OR REPLACE INTO moments (
            id, timestamp, duration_seconds, camera_id, location_label,
            clip_path, thumbnail_path, audio_path,
            technical_score, activity_score, interest_score, farm_vibe_score, overall_score,
            tags, animals_present, emotional_register,
            narrative_note, suggested_use, conditions,
            passed_gates, reviewed, used_in_content, archived
        ) VALUES (
            :id, :timestamp, :duration_seconds, :camera_id, :location_label,
            :clip_path, :thumbnail_path, :audio_path,
            :technical_score, :activity_score, :interest_score, :farm_vibe_score, :overall_score,
            :tags, :animals_present, :emotional_register,
            :narrative_note, :suggested_use, :conditions,
            :passed_gates, :reviewed, :used_in_content, :archived
        )
    """
    with db._connect() as conn:
        conn.execute(sql, row)
    logger.debug("Saved moment %s (passed_gates=%s, overall_score=%.2f)",
                 moment.id, moment.passed_gates, moment.overall_score)


def get_moment(db: Database, moment_id: str) -> Optional[Moment]:
    """Fetch a single Moment by ID. Returns None if not found."""
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM moments WHERE id = ?", (moment_id,)).fetchone()
    return _row_to_moment(row) if row else None


def get_moments_today(db: Database) -> List[Moment]:
    """Return all Moments with a timestamp on today's UTC date, sorted by overall_score desc."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM moments WHERE date(timestamp) = date('now') ORDER BY overall_score DESC"
        ).fetchall()
    return [_row_to_moment(r) for r in rows]


def get_moments_passed(db: Database, limit: int = 100) -> List[Moment]:
    """Return Moments that passed all gates and have not yet been used in content."""
    with db._connect() as conn:
        rows = conn.execute(
            """SELECT * FROM moments
               WHERE passed_gates = 1
                 AND used_in_content = '[]'
                 AND archived = 0
               ORDER BY overall_score DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [_row_to_moment(r) for r in rows]


def mark_moment_reviewed(db: Database, moment_id: str) -> None:
    with db._connect() as conn:
        conn.execute("UPDATE moments SET reviewed = 1 WHERE id = ?", (moment_id,))


def mark_moment_used(db: Database, moment_id: str, content_id: str) -> None:
    """Append content_id to the moment's used_in_content list."""
    moment = get_moment(db, moment_id)
    if moment is None:
        return
    if content_id not in moment.used_in_content:
        moment.used_in_content.append(content_id)
        with db._connect() as conn:
            conn.execute(
                "UPDATE moments SET used_in_content = ? WHERE id = ?",
                (_to_json(moment.used_in_content), moment_id),
            )


# ─── ANIMAL CRUD ───────────────────────────────────────────────────────────────

def _animal_to_row(a: Animal) -> dict:
    return {
        "id": a.id,
        "name": a.name,
        "group_name": a.group,
        "type": a.type,
        "markings": a.markings,
        "temperament": a.temperament,
        "arrival_date": _date_to_str(a.arrival_date),
        "origin": a.origin,
        "current_phase": a.current_phase,
        "harvest_date": _date_to_str(a.harvest_date),
        "moment_appearances": _to_json(a.moment_appearances),
        "milestones": _to_json(a.milestones),
    }


def _row_to_animal(row: sqlite3.Row) -> Animal:
    return Animal(
        id=row["id"],
        name=row["name"],
        group=row["group_name"],
        type=row["type"],
        markings=row["markings"],
        temperament=row["temperament"],
        arrival_date=_str_to_date(row["arrival_date"]),
        origin=row["origin"],
        current_phase=row["current_phase"],
        harvest_date=_str_to_date(row["harvest_date"]),
        moment_appearances=_from_json(row["moment_appearances"], []),
        milestones=_from_json(row["milestones"], []),
    )


def save_animal(db: Database, animal: Animal) -> None:
    """Insert or replace an Animal record."""
    row = _animal_to_row(animal)
    sql = """
        INSERT OR REPLACE INTO animals (
            id, name, group_name, type, markings, temperament,
            arrival_date, origin, current_phase, harvest_date,
            moment_appearances, milestones,
            updated_at
        ) VALUES (
            :id, :name, :group_name, :type, :markings, :temperament,
            :arrival_date, :origin, :current_phase, :harvest_date,
            :moment_appearances, :milestones,
            strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        )
    """
    with db._connect() as conn:
        conn.execute(sql, row)


def get_animal(db: Database, animal_id: str) -> Optional[Animal]:
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM animals WHERE id = ?", (animal_id,)).fetchone()
    return _row_to_animal(row) if row else None


def get_all_animals(db: Database) -> List[Animal]:
    with db._connect() as conn:
        rows = conn.execute("SELECT * FROM animals ORDER BY name").fetchall()
    return [_row_to_animal(r) for r in rows]


# ─── CONTENT CRUD ──────────────────────────────────────────────────────────────

def _content_to_row(c: Content) -> dict:
    return {
        "id": c.id,
        "created_at": _datetime_to_str(c.created_at),
        "type": c.type,
        "format": c.format,
        "source_moment_ids": _to_json(c.source_moment_ids),
        "source_animal_ids": _to_json(c.source_animal_ids),
        "media_path": c.media_path,
        "caption": c.caption,
        "hook": c.hook,
        "hashtags": _to_json(c.hashtags),
        "emotional_tone": c.emotional_tone,
        "farm_narrative_arc": c.farm_narrative_arc,
        "cta_included": int(c.cta_included),
        "product_referenced": c.product_referenced,
        "link": c.link,
        "platforms": _to_json(c.platforms),
        "scheduled_for": _datetime_to_str(c.scheduled_for),
        "status": c.status,
        "performance": _to_json(c.performance),
    }


def _row_to_content(row: sqlite3.Row) -> Content:
    return Content(
        id=row["id"],
        created_at=_str_to_datetime(row["created_at"]),
        type=row["type"],
        format=row["format"],
        source_moment_ids=_from_json(row["source_moment_ids"], []),
        source_animal_ids=_from_json(row["source_animal_ids"], []),
        media_path=row["media_path"],
        caption=row["caption"],
        hook=row["hook"],
        hashtags=_from_json(row["hashtags"], []),
        emotional_tone=row["emotional_tone"],
        farm_narrative_arc=row["farm_narrative_arc"],
        cta_included=bool(row["cta_included"]),
        product_referenced=row["product_referenced"],
        link=row["link"],
        platforms=_from_json(row["platforms"], []),
        scheduled_for=_str_to_datetime(row["scheduled_for"]),
        status=row["status"],
        performance=_from_json(row["performance"], {}),
    )


def save_content(db: Database, content: Content) -> None:
    """Insert or replace a Content record."""
    row = _content_to_row(content)
    sql = """
        INSERT OR REPLACE INTO content (
            id, created_at, type, format,
            source_moment_ids, source_animal_ids,
            media_path, caption, hook, hashtags,
            emotional_tone, farm_narrative_arc,
            cta_included, product_referenced, link,
            platforms, scheduled_for, status, performance
        ) VALUES (
            :id, :created_at, :type, :format,
            :source_moment_ids, :source_animal_ids,
            :media_path, :caption, :hook, :hashtags,
            :emotional_tone, :farm_narrative_arc,
            :cta_included, :product_referenced, :link,
            :platforms, :scheduled_for, :status, :performance
        )
    """
    with db._connect() as conn:
        conn.execute(sql, row)


def get_content(db: Database, content_id: str) -> Optional[Content]:
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM content WHERE id = ?", (content_id,)).fetchone()
    return _row_to_content(row) if row else None


def get_content_drafts(db: Database) -> List[Content]:
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM content WHERE status = 'draft' ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_content(r) for r in rows]


# ─── PERSON CRUD ───────────────────────────────────────────────────────────────

def _person_to_row(p: Person) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "email": p.email,
        "phone": p.phone,
        "location": p.location,
        "type": p.type,
        "first_contact": _date_to_str(p.first_contact),
        "channel": p.channel,
        "warmth": p.warmth,
        "purchases": _to_json(p.purchases),
        "content_engaged": _to_json(p.content_engaged),
        "conversations": _to_json(p.conversations),
        "visited_farm": int(p.visited_farm),
        "preferred_products": _to_json(p.preferred_products),
        "communication_cadence": p.communication_cadence,
        "next_action_type": p.next_action_type,
        "next_action_reason": p.next_action_reason,
        "next_action_content": p.next_action_content,
    }


def _row_to_person(row: sqlite3.Row) -> Person:
    return Person(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        phone=row["phone"],
        location=row["location"],
        type=row["type"],
        first_contact=_str_to_date(row["first_contact"]),
        channel=row["channel"],
        warmth=row["warmth"],
        purchases=_from_json(row["purchases"], []),
        content_engaged=_from_json(row["content_engaged"], []),
        conversations=_from_json(row["conversations"], []),
        visited_farm=bool(row["visited_farm"]),
        preferred_products=_from_json(row["preferred_products"], []),
        communication_cadence=row["communication_cadence"],
        next_action_type=row["next_action_type"],
        next_action_reason=row["next_action_reason"],
        next_action_content=row["next_action_content"],
    )


def save_person(db: Database, person: Person) -> None:
    """Insert or replace a Person record."""
    row = _person_to_row(person)
    sql = """
        INSERT OR REPLACE INTO persons (
            id, name, email, phone, location, type,
            first_contact, channel, warmth,
            purchases, content_engaged, conversations,
            visited_farm, preferred_products,
            communication_cadence,
            next_action_type, next_action_reason, next_action_content,
            updated_at
        ) VALUES (
            :id, :name, :email, :phone, :location, :type,
            :first_contact, :channel, :warmth,
            :purchases, :content_engaged, :conversations,
            :visited_farm, :preferred_products,
            :communication_cadence,
            :next_action_type, :next_action_reason, :next_action_content,
            strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        )
    """
    with db._connect() as conn:
        conn.execute(sql, row)


def get_person(db: Database, person_id: str) -> Optional[Person]:
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    return _row_to_person(row) if row else None


def get_persons_with_pending_actions(db: Database) -> List[Person]:
    """Return persons where next_action_type is not 'none'."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM persons WHERE next_action_type != 'none' ORDER BY warmth DESC"
        ).fetchall()
    return [_row_to_person(r) for r in rows]


def get_all_persons(db: Database) -> List[Person]:
    with db._connect() as conn:
        rows = conn.execute("SELECT * FROM persons ORDER BY warmth DESC").fetchall()
    return [_row_to_person(r) for r in rows]
