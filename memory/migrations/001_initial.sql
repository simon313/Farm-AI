-- memory/migrations/001_initial.sql
-- Initial schema for Bardo Farm SQLite database.
-- All list and dict fields are stored as JSON strings.
-- Run once at startup via db.py if tables do not exist.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;


-- ─── MOMENTS ──────────────────────────────────────────────────────────────────
-- The atomic unit of the system. One row per clip that passed all four gates
-- (or was recorded but failed — passed_gates distinguishes them).

CREATE TABLE IF NOT EXISTS moments (
    id                  TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,          -- ISO 8601 UTC
    duration_seconds    REAL NOT NULL DEFAULT 0.0,
    camera_id           TEXT NOT NULL DEFAULT '',
    location_label      TEXT NOT NULL DEFAULT '',
    clip_path           TEXT NOT NULL DEFAULT '',
    thumbnail_path      TEXT NOT NULL DEFAULT '',
    audio_path          TEXT,                   -- NULL if no audio

    -- Gate scores
    technical_score     REAL NOT NULL DEFAULT 0.0,
    activity_score      REAL NOT NULL DEFAULT 0.0,
    interest_score      REAL NOT NULL DEFAULT 0.0,
    farm_vibe_score     REAL NOT NULL DEFAULT 0.0,
    overall_score       REAL NOT NULL DEFAULT 0.0,

    -- Editorial metadata (JSON strings)
    tags                TEXT NOT NULL DEFAULT '[]',
    animals_present     TEXT NOT NULL DEFAULT '[]',
    emotional_register  TEXT NOT NULL DEFAULT '[]',
    narrative_note      TEXT NOT NULL DEFAULT '',
    suggested_use       TEXT NOT NULL DEFAULT '',
    conditions          TEXT NOT NULL DEFAULT '{}',

    -- Status
    passed_gates        INTEGER NOT NULL DEFAULT 0,   -- SQLite boolean: 0/1
    reviewed            INTEGER NOT NULL DEFAULT 0,
    used_in_content     TEXT NOT NULL DEFAULT '[]',   -- JSON list of content IDs
    archived            INTEGER NOT NULL DEFAULT 0,

    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_moments_timestamp    ON moments (timestamp);
CREATE INDEX IF NOT EXISTS idx_moments_passed_gates ON moments (passed_gates);
CREATE INDEX IF NOT EXISTS idx_moments_overall_score ON moments (overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_moments_camera_id    ON moments (camera_id);


-- ─── ANIMALS ──────────────────────────────────────────────────────────────────
-- Individual pigs. Characters the audience follows.
-- Seeded from config/animals.yaml at startup.

CREATE TABLE IF NOT EXISTS animals (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL DEFAULT '',
    group_name          TEXT NOT NULL DEFAULT '',   -- 'group' is a SQL keyword
    type                TEXT NOT NULL DEFAULT '',
    markings            TEXT NOT NULL DEFAULT '',
    temperament         TEXT NOT NULL DEFAULT '',
    arrival_date        TEXT,                       -- ISO 8601 date string or NULL
    origin              TEXT NOT NULL DEFAULT '',
    current_phase       TEXT NOT NULL DEFAULT '',
    harvest_date        TEXT,                       -- ISO 8601 date string or NULL
    moment_appearances  TEXT NOT NULL DEFAULT '[]', -- JSON list of moment IDs
    milestones          TEXT NOT NULL DEFAULT '[]', -- JSON list of dicts

    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_animals_group ON animals (group_name);
CREATE INDEX IF NOT EXISTS idx_animals_phase ON animals (current_phase);


-- ─── CONTENT ──────────────────────────────────────────────────────────────────
-- Assembled content created by producer and storyteller agents.
-- Moves through draft -> approved -> posted lifecycle.

CREATE TABLE IF NOT EXISTS content (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,              -- ISO 8601 UTC
    type                TEXT NOT NULL DEFAULT '',   -- post, reel, story, email, dispatch
    format              TEXT NOT NULL DEFAULT '',   -- video, image, text, carousel
    source_moment_ids   TEXT NOT NULL DEFAULT '[]',
    source_animal_ids   TEXT NOT NULL DEFAULT '[]',
    media_path          TEXT NOT NULL DEFAULT '',
    caption             TEXT NOT NULL DEFAULT '',
    hook                TEXT NOT NULL DEFAULT '',
    hashtags            TEXT NOT NULL DEFAULT '[]',
    emotional_tone      TEXT NOT NULL DEFAULT '',
    farm_narrative_arc  TEXT NOT NULL DEFAULT '',
    cta_included        INTEGER NOT NULL DEFAULT 0,
    product_referenced  TEXT,
    link                TEXT,
    platforms           TEXT NOT NULL DEFAULT '[]',
    scheduled_for       TEXT,                       -- ISO 8601 UTC or NULL
    status              TEXT NOT NULL DEFAULT 'draft',
    performance         TEXT NOT NULL DEFAULT '{}'  -- JSON dict per platform
);

CREATE INDEX IF NOT EXISTS idx_content_status       ON content (status);
CREATE INDEX IF NOT EXISTS idx_content_created_at   ON content (created_at);
CREATE INDEX IF NOT EXISTS idx_content_scheduled_for ON content (scheduled_for);


-- ─── PERSONS ──────────────────────────────────────────────────────────────────
-- Everyone the farm has a relationship with.

CREATE TABLE IF NOT EXISTS persons (
    id                      TEXT PRIMARY KEY,
    name                    TEXT NOT NULL DEFAULT '',
    email                   TEXT,
    phone                   TEXT,
    location                TEXT,
    type                    TEXT NOT NULL DEFAULT '',   -- customer, follower, prospect, press
    first_contact           TEXT,                       -- ISO 8601 date string or NULL
    channel                 TEXT NOT NULL DEFAULT '',
    warmth                  REAL NOT NULL DEFAULT 0.0,
    purchases               TEXT NOT NULL DEFAULT '[]',
    content_engaged         TEXT NOT NULL DEFAULT '[]',
    conversations           TEXT NOT NULL DEFAULT '[]',
    visited_farm            INTEGER NOT NULL DEFAULT 0,
    preferred_products      TEXT NOT NULL DEFAULT '[]',
    communication_cadence   TEXT NOT NULL DEFAULT 'monthly',
    next_action_type        TEXT NOT NULL DEFAULT 'none',
    next_action_reason      TEXT NOT NULL DEFAULT '',
    next_action_content     TEXT NOT NULL DEFAULT '',

    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_persons_type     ON persons (type);
CREATE INDEX IF NOT EXISTS idx_persons_warmth   ON persons (warmth DESC);
CREATE INDEX IF NOT EXISTS idx_persons_email    ON persons (email);
