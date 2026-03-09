"""
config/loader.py
Loads and validates all farm configuration from the three YAML files.

Usage:
    from config.loader import load_config, get_config

    # At application startup:
    load_config("config/")

    # Anywhere else in the codebase:
    cfg = get_config()
    cameras = cfg.cameras
    gate1 = cfg.gate1

Call load_config() exactly once at startup. All subsequent calls to get_config()
return the same validated Config object without re-reading files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ─── CONFIG DATACLASSES ────────────────────────────────────────────────────────


@dataclass
class CameraConfig:
    id: str
    ip: str
    port: int
    label: str
    notes: str = ""

    def rtsp_url(self, username: str, password: str) -> str:
        return f"rtsp://{username}:{password}@{self.ip}:{self.port}/h264Preview_01_main"


@dataclass
class AnimalConfig:
    id: str
    name: str
    group: str
    type: str
    markings: str
    temperament: str
    arrival_date: str
    origin: str
    current_phase: str


@dataclass
class Gate1Config:
    min_brightness: int
    min_sharpness: float
    min_duration_seconds: int
    max_shake_threshold: float
    audio_quality_check: bool


@dataclass
class Gate2Config:
    min_motion_area_ratio: float
    min_motion_frames: int
    background_history: int
    background_var_threshold: int


@dataclass
class Gate3Config:
    min_register_score: float
    thumbnail_frames: int


@dataclass
class Gate4Config:
    min_vibe_score: float


@dataclass
class MotionConfig:
    sensitivity: str
    clip_pre_buffer_seconds: int
    clip_post_buffer_seconds: int
    max_clip_duration_seconds: int


@dataclass
class StorageConfig:
    clips_dir: str
    thumbnails_dir: str
    content_dir: str
    database_path: str
    retain_failed_clips: bool


@dataclass
class FarmIdentityConfig:
    name: str
    location: str
    what_we_raise: str
    voice: str
    what_we_believe: str


@dataclass
class Config:
    """
    Fully validated, parsed configuration for the entire system.
    Constructed once by load_config() and shared via get_config().
    """
    cameras: List[CameraConfig]
    animals: List[AnimalConfig]
    farm: FarmIdentityConfig
    season: str
    current_narrative_arc: str
    products_available: List[str]
    products_coming: List[Dict[str, str]]
    gate1: Gate1Config
    gate2: Gate2Config
    gate3: Gate3Config
    gate4: Gate4Config
    motion: MotionConfig
    storage: StorageConfig

    def camera_by_id(self, camera_id: str) -> Optional[CameraConfig]:
        """Return the CameraConfig with the given id, or None."""
        return next((c for c in self.cameras if c.id == camera_id), None)

    def animal_by_id(self, animal_id: str) -> Optional[AnimalConfig]:
        """Return the AnimalConfig with the given id, or None."""
        return next((a for a in self.animals if a.id == animal_id), None)


# ─── SINGLETON ─────────────────────────────────────────────────────────────────

_config: Optional[Config] = None


def get_config() -> Config:
    """Return the loaded Config. Raises RuntimeError if load_config() has not been called."""
    if _config is None:
        raise RuntimeError(
            "Config has not been loaded. Call load_config(config_dir) at application startup."
        )
    return _config


# ─── LOADER ────────────────────────────────────────────────────────────────────

def load_config(config_dir: str = "config") -> Config:
    """
    Read and validate all three YAML config files. Store result in the module singleton.
    Raises ConfigError with a clear message if any required field is missing or invalid.
    Call this exactly once at application startup.
    """
    global _config
    base = Path(config_dir)

    cameras_raw = _load_yaml(base / "cameras.yaml")
    animals_raw = _load_yaml(base / "animals.yaml")
    farm_raw = _load_yaml(base / "farm.yaml")

    cameras = _parse_cameras(cameras_raw)
    animals = _parse_animals(animals_raw)
    farm_identity, farm_operational = _parse_farm(farm_raw)

    _config = Config(
        cameras=cameras,
        animals=animals,
        farm=farm_identity,
        season=farm_operational["season"],
        current_narrative_arc=farm_operational["current_narrative_arc"],
        products_available=farm_operational["products_available"],
        products_coming=farm_operational["products_coming"],
        gate1=farm_operational["gate1"],
        gate2=farm_operational["gate2"],
        gate3=farm_operational["gate3"],
        gate4=farm_operational["gate4"],
        motion=farm_operational["motion"],
        storage=farm_operational["storage"],
    )
    return _config


# ─── INTERNAL PARSERS ──────────────────────────────────────────────────────────

class ConfigError(Exception):
    """Raised when a required config field is missing or has an invalid value."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file is empty or not a YAML mapping: {path}")
    return data


def _require(data: dict, key: str, context: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigError(f"Missing required field '{key}' in {context}")
    return data[key]


def _parse_cameras(raw: dict) -> List[CameraConfig]:
    entries = _require(raw, "cameras", "cameras.yaml")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ConfigError("cameras.yaml must contain at least one camera under 'cameras:'")

    cameras = []
    for i, entry in enumerate(entries):
        ctx = f"cameras.yaml cameras[{i}]"
        cameras.append(CameraConfig(
            id=str(_require(entry, "id", ctx)),
            ip=str(_require(entry, "ip", ctx)),
            port=int(_require(entry, "port", ctx)),
            label=str(_require(entry, "label", ctx)),
            notes=str(entry.get("notes", "")),
        ))

    ids = [c.id for c in cameras]
    if len(ids) != len(set(ids)):
        raise ConfigError("cameras.yaml contains duplicate camera IDs")

    return cameras


def _parse_animals(raw: dict) -> List[AnimalConfig]:
    entries = _require(raw, "animals", "animals.yaml")
    if not isinstance(entries, list):
        raise ConfigError("animals.yaml must contain a list under 'animals:'")

    animals = []
    for i, entry in enumerate(entries):
        ctx = f"animals.yaml animals[{i}]"
        animals.append(AnimalConfig(
            id=str(_require(entry, "id", ctx)),
            name=str(_require(entry, "name", ctx)),
            group=str(_require(entry, "group", ctx)),
            type=str(_require(entry, "type", ctx)),
            markings=str(_require(entry, "markings", ctx)),
            temperament=str(_require(entry, "temperament", ctx)),
            arrival_date=str(_require(entry, "arrival_date", ctx)),
            origin=str(_require(entry, "origin", ctx)),
            current_phase=str(_require(entry, "current_phase", ctx)),
        ))
    return animals


def _parse_farm(raw: dict) -> tuple:
    """Returns (FarmIdentityConfig, dict of operational settings)."""
    farm_block = _require(raw, "farm", "farm.yaml")
    ctx = "farm.yaml farm"
    identity = FarmIdentityConfig(
        name=str(_require(farm_block, "name", ctx)),
        location=str(_require(farm_block, "location", ctx)),
        what_we_raise=str(_require(farm_block, "what_we_raise", ctx)),
        voice=str(_require(farm_block, "voice", ctx)),
        what_we_believe=str(_require(farm_block, "what_we_believe", ctx)),
    )

    season = str(_require(raw, "season", "farm.yaml"))
    arc = str(_require(raw, "current_narrative_arc", "farm.yaml"))
    products_available = _require(raw, "products_available", "farm.yaml")
    products_coming = raw.get("products_coming", [])

    g1 = _require(raw, "gate1", "farm.yaml")
    gate1 = Gate1Config(
        min_brightness=int(_require(g1, "min_brightness", "farm.yaml gate1")),
        min_sharpness=float(_require(g1, "min_sharpness", "farm.yaml gate1")),
        min_duration_seconds=int(_require(g1, "min_duration_seconds", "farm.yaml gate1")),
        max_shake_threshold=float(_require(g1, "max_shake_threshold", "farm.yaml gate1")),
        audio_quality_check=bool(_require(g1, "audio_quality_check", "farm.yaml gate1")),
    )

    g2 = _require(raw, "gate2", "farm.yaml")
    gate2 = Gate2Config(
        min_motion_area_ratio=float(_require(g2, "min_motion_area_ratio", "farm.yaml gate2")),
        min_motion_frames=int(_require(g2, "min_motion_frames", "farm.yaml gate2")),
        background_history=int(_require(g2, "background_history", "farm.yaml gate2")),
        background_var_threshold=int(_require(g2, "background_var_threshold", "farm.yaml gate2")),
    )

    g3 = _require(raw, "gate3", "farm.yaml")
    gate3 = Gate3Config(
        min_register_score=float(_require(g3, "min_register_score", "farm.yaml gate3")),
        thumbnail_frames=int(_require(g3, "thumbnail_frames", "farm.yaml gate3")),
    )

    g4 = _require(raw, "gate4", "farm.yaml")
    gate4 = Gate4Config(
        min_vibe_score=float(_require(g4, "min_vibe_score", "farm.yaml gate4")),
    )

    mot = _require(raw, "motion", "farm.yaml")
    motion = MotionConfig(
        sensitivity=str(_require(mot, "sensitivity", "farm.yaml motion")),
        clip_pre_buffer_seconds=int(_require(mot, "clip_pre_buffer_seconds", "farm.yaml motion")),
        clip_post_buffer_seconds=int(_require(mot, "clip_post_buffer_seconds", "farm.yaml motion")),
        max_clip_duration_seconds=int(_require(mot, "max_clip_duration_seconds", "farm.yaml motion")),
    )

    stor = _require(raw, "storage", "farm.yaml")
    storage = StorageConfig(
        clips_dir=str(_require(stor, "clips_dir", "farm.yaml storage")),
        thumbnails_dir=str(_require(stor, "thumbnails_dir", "farm.yaml storage")),
        content_dir=str(_require(stor, "content_dir", "farm.yaml storage")),
        database_path=str(_require(stor, "database_path", "farm.yaml storage")),
        retain_failed_clips=bool(_require(stor, "retain_failed_clips", "farm.yaml storage")),
    )

    operational = {
        "season": season,
        "current_narrative_arc": arc,
        "products_available": products_available,
        "products_coming": products_coming,
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
        "gate4": gate4,
        "motion": motion,
        "storage": storage,
    }
    return identity, operational
