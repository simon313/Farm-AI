"""
tests/test_models.py
Tests for dataclass serialization to/from SQLite via memory/db.py.
Uses an in-memory SQLite database — no files written to disk.
"""

import pytest
from datetime import date, datetime

from memory.db import (
    Database,
    get_animal,
    get_content,
    get_content_drafts,
    get_moment,
    get_moments_passed,
    get_moments_today,
    get_person,
    mark_moment_reviewed,
    mark_moment_used,
    save_animal,
    save_content,
    save_moment,
    save_person,
)
from memory.models import Animal, Content, Moment, Person


@pytest.fixture
def db(tmp_path):
    """Provide a fresh Database backed by a temp file for each test."""
    return Database(str(tmp_path / "test.db"))


# ─── MOMENT ────────────────────────────────────────────────────────────────────

class TestMoment:
    def _sample(self) -> Moment:
        return Moment(
            id="moment-001",
            timestamp=datetime(2025, 3, 9, 8, 0, 0),
            duration_seconds=12.5,
            camera_id="north_pen_cam",
            location_label="North Pen",
            clip_path="/storage/clips/moment-001.mp4",
            thumbnail_path="/storage/thumbnails/moment-001.jpg",
            audio_path=None,
            technical_score=0.85,
            activity_score=0.72,
            interest_score=0.91,
            farm_vibe_score=0.88,
            overall_score=0.84,
            tags=["foraging", "social"],
            animals_present=["earl"],
            emotional_register=["warmth", "curiosity"],
            narrative_note="Earl investigates a fresh mud wallow with characteristic deliberateness.",
            suggested_use="reel",
            conditions={"light": "morning", "weather": "clear", "time_of_day": "morning"},
            passed_gates=True,
            reviewed=False,
            used_in_content=[],
            archived=False,
        )

    def test_round_trip(self, db):
        m = self._sample()
        save_moment(db, m)
        fetched = get_moment(db, m.id)
        assert fetched is not None
        assert fetched.id == m.id
        assert fetched.camera_id == m.camera_id
        assert fetched.duration_seconds == m.duration_seconds
        assert fetched.tags == m.tags
        assert fetched.animals_present == m.animals_present
        assert fetched.emotional_register == m.emotional_register
        assert fetched.conditions == m.conditions
        assert fetched.passed_gates is True
        assert fetched.reviewed is False
        assert fetched.audio_path is None
        assert fetched.overall_score == pytest.approx(0.84)

    def test_missing_returns_none(self, db):
        assert get_moment(db, "nonexistent") is None

    def test_upsert_updates_fields(self, db):
        m = self._sample()
        save_moment(db, m)
        m.narrative_note = "Updated note."
        m.overall_score = 0.99
        save_moment(db, m)
        fetched = get_moment(db, m.id)
        assert fetched.narrative_note == "Updated note."
        assert fetched.overall_score == pytest.approx(0.99)

    def test_mark_reviewed(self, db):
        m = self._sample()
        save_moment(db, m)
        mark_moment_reviewed(db, m.id)
        assert get_moment(db, m.id).reviewed is True

    def test_mark_used(self, db):
        m = self._sample()
        save_moment(db, m)
        mark_moment_used(db, m.id, "content-abc")
        fetched = get_moment(db, m.id)
        assert "content-abc" in fetched.used_in_content

    def test_mark_used_no_duplicates(self, db):
        m = self._sample()
        save_moment(db, m)
        mark_moment_used(db, m.id, "content-abc")
        mark_moment_used(db, m.id, "content-abc")
        assert get_moment(db, m.id).used_in_content.count("content-abc") == 1

    def test_get_moments_passed_excludes_used(self, db):
        m = self._sample()
        save_moment(db, m)
        assert len(get_moments_passed(db)) == 1
        mark_moment_used(db, m.id, "content-xyz")
        # used_in_content is now non-empty — should be excluded
        assert len(get_moments_passed(db)) == 0

    def test_get_moments_passed_excludes_failed(self, db):
        m = self._sample()
        m.passed_gates = False
        save_moment(db, m)
        assert get_moments_passed(db) == []


# ─── ANIMAL ────────────────────────────────────────────────────────────────────

class TestAnimal:
    def _sample(self) -> Animal:
        return Animal(
            id="animal-earl",
            name="Earl",
            group="north_pen",
            type="boar",
            markings="Large dark grey, notch in left ear",
            temperament="dominant",
            arrival_date=date(2024, 3, 15),
            origin="Bardo Farm born",
            current_phase="finishing",
            harvest_date=None,
            moment_appearances=["moment-001"],
            milestones=[{"date": "2025-01-01", "description": "moved to finishing pen"}],
        )

    def test_round_trip(self, db):
        a = self._sample()
        save_animal(db, a)
        fetched = get_animal(db, a.id)
        assert fetched is not None
        assert fetched.id == a.id
        assert fetched.name == "Earl"
        assert fetched.group == "north_pen"
        assert fetched.arrival_date == date(2024, 3, 15)
        assert fetched.harvest_date is None
        assert fetched.moment_appearances == ["moment-001"]
        assert len(fetched.milestones) == 1
        assert fetched.milestones[0]["description"] == "moved to finishing pen"

    def test_missing_returns_none(self, db):
        assert get_animal(db, "nonexistent") is None


# ─── CONTENT ───────────────────────────────────────────────────────────────────

class TestContent:
    def _sample(self) -> Content:
        return Content(
            id="content-001",
            created_at=datetime(2025, 3, 9, 12, 0, 0),
            type="reel",
            format="video",
            source_moment_ids=["moment-001"],
            source_animal_ids=["animal-earl"],
            media_path="/storage/content/content-001.mp4",
            caption="Earl found the mud. As he does.",
            hook="He found it.",
            hashtags=["#bardofarm", "#woodsraisedpork"],
            emotional_tone="warm",
            farm_narrative_arc="finishing season",
            cta_included=True,
            product_referenced="smoked shoulder",
            link=None,
            platforms=["instagram"],
            scheduled_for=None,
            status="draft",
            performance={},
        )

    def test_round_trip(self, db):
        c = self._sample()
        save_content(db, c)
        fetched = get_content(db, c.id)
        assert fetched is not None
        assert fetched.caption == "Earl found the mud. As he does."
        assert fetched.hashtags == ["#bardofarm", "#woodsraisedpork"]
        assert fetched.cta_included is True
        assert fetched.link is None
        assert fetched.status == "draft"
        assert fetched.platforms == ["instagram"]

    def test_get_drafts(self, db):
        c = self._sample()
        save_content(db, c)
        drafts = get_content_drafts(db)
        assert len(drafts) == 1
        assert drafts[0].id == c.id

    def test_approved_not_in_drafts(self, db):
        c = self._sample()
        c.status = "approved"
        save_content(db, c)
        assert get_content_drafts(db) == []

    def test_missing_returns_none(self, db):
        assert get_content(db, "nonexistent") is None


# ─── PERSON ────────────────────────────────────────────────────────────────────

class TestPerson:
    def _sample(self) -> Person:
        return Person(
            id="person-001",
            name="Jane Doe",
            email="jane@example.com",
            phone=None,
            location="Virginia",
            type="customer",
            first_contact=date(2024, 11, 1),
            channel="instagram",
            warmth=0.75,
            purchases=[{"date": "2024-12-01", "product": "smoked shoulder", "amount": 45.0}],
            content_engaged=["content-001"],
            conversations=[],
            visited_farm=True,
            preferred_products=["smoked shoulder", "lardo"],
            communication_cadence="monthly",
            next_action_type="offer",
            next_action_reason="Has not purchased since December, spring cuts available.",
            next_action_content="Hey Jane — spring cuts are ready...",
        )

    def test_round_trip(self, db):
        p = self._sample()
        save_person(db, p)
        fetched = get_person(db, p.id)
        assert fetched is not None
        assert fetched.name == "Jane Doe"
        assert fetched.email == "jane@example.com"
        assert fetched.warmth == pytest.approx(0.75)
        assert fetched.visited_farm is True
        assert len(fetched.purchases) == 1
        assert fetched.purchases[0]["product"] == "smoked shoulder"
        assert fetched.preferred_products == ["smoked shoulder", "lardo"]
        assert fetched.next_action_type == "offer"

    def test_missing_returns_none(self, db):
        assert get_person(db, "nonexistent") is None

    def test_upsert_updates_warmth(self, db):
        p = self._sample()
        save_person(db, p)
        p.warmth = 0.9
        save_person(db, p)
        assert get_person(db, p.id).warmth == pytest.approx(0.9)
