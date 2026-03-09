"""
tests/test_config.py
Tests for config loading and validation.
Uses synthetic YAML written to tmp_path for failure cases,
and the real config/ directory for a smoke test.
"""

import pytest
import yaml
from pathlib import Path

from config.loader import ConfigError, load_config, get_config


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


def _minimal_cameras() -> dict:
    return {
        "cameras": [
            {"id": "cam1", "ip": "10.0.0.1", "port": 554, "label": "Test Cam"}
        ]
    }


def _minimal_animals() -> dict:
    return {
        "animals": [
            {
                "id": "pig1", "name": "Piggy", "group": "north_pen",
                "type": "boar", "markings": "big", "temperament": "curious",
                "arrival_date": "2024-01-01", "origin": "local",
                "current_phase": "growing",
            }
        ]
    }


def _minimal_farm() -> dict:
    return {
        "farm": {
            "name": "Test Farm", "location": "Virginia",
            "what_we_raise": "pork", "voice": "honest",
            "what_we_believe": "Pigs belong outside.",
        },
        "season": "spring",
        "current_narrative_arc": "test arc",
        "products_available": ["smoked shoulder"],
        "products_coming": [],
        "gate1": {
            "min_brightness": 30, "min_sharpness": 80.0,
            "min_duration_seconds": 4, "max_shake_threshold": 15.0,
            "audio_quality_check": True,
        },
        "gate2": {
            "min_motion_area_ratio": 0.02, "min_motion_frames": 5,
            "background_history": 500, "background_var_threshold": 16,
        },
        "gate3": {"min_register_score": 0.6, "thumbnail_frames": 3},
        "gate4": {"min_vibe_score": 0.7},
        "motion": {
            "sensitivity": "medium", "clip_pre_buffer_seconds": 3,
            "clip_post_buffer_seconds": 10, "max_clip_duration_seconds": 60,
        },
        "storage": {
            "clips_dir": "storage/clips", "thumbnails_dir": "storage/thumbnails",
            "content_dir": "storage/content", "database_path": "storage/test.db",
            "retain_failed_clips": False,
        },
    }


@pytest.fixture
def config_dir(tmp_path):
    """Write minimal valid YAML files to a temp directory and return the path string."""
    _write_yaml(tmp_path / "cameras.yaml", _minimal_cameras())
    _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
    _write_yaml(tmp_path / "farm.yaml", _minimal_farm())
    return str(tmp_path)


# ─── HAPPY PATH ────────────────────────────────────────────────────────────────

class TestLoadConfigHappyPath:
    def test_loads_without_error(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg is not None

    def test_get_config_returns_same_object(self, config_dir):
        cfg = load_config(config_dir)
        assert get_config() is cfg

    def test_cameras_parsed(self, config_dir):
        cfg = load_config(config_dir)
        assert len(cfg.cameras) == 1
        assert cfg.cameras[0].id == "cam1"
        assert cfg.cameras[0].ip == "10.0.0.1"
        assert cfg.cameras[0].port == 554
        assert cfg.cameras[0].label == "Test Cam"

    def test_camera_rtsp_url(self, config_dir):
        cfg = load_config(config_dir)
        url = cfg.cameras[0].rtsp_url("admin", "secret")
        assert url == "rtsp://admin:secret@10.0.0.1:554/h264Preview_01_main"

    def test_animals_parsed(self, config_dir):
        cfg = load_config(config_dir)
        assert len(cfg.animals) == 1
        assert cfg.animals[0].id == "pig1"
        assert cfg.animals[0].name == "Piggy"

    def test_farm_identity_parsed(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.farm.name == "Test Farm"
        assert cfg.farm.voice == "honest"
        assert "Pigs belong outside" in cfg.farm.what_we_believe

    def test_season_and_arc(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.season == "spring"
        assert cfg.current_narrative_arc == "test arc"

    def test_gate1_thresholds(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.gate1.min_brightness == 30
        assert cfg.gate1.min_sharpness == 80.0
        assert cfg.gate1.min_duration_seconds == 4
        assert cfg.gate1.audio_quality_check is True

    def test_gate2_thresholds(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.gate2.min_motion_area_ratio == pytest.approx(0.02)
        assert cfg.gate2.min_motion_frames == 5

    def test_gate3_thresholds(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.gate3.min_register_score == pytest.approx(0.6)
        assert cfg.gate3.thumbnail_frames == 3

    def test_gate4_thresholds(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.gate4.min_vibe_score == pytest.approx(0.7)

    def test_motion_config(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.motion.sensitivity == "medium"
        assert cfg.motion.max_clip_duration_seconds == 60

    def test_storage_config(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.storage.clips_dir == "storage/clips"
        assert cfg.storage.retain_failed_clips is False

    def test_camera_by_id_found(self, config_dir):
        cfg = load_config(config_dir)
        cam = cfg.camera_by_id("cam1")
        assert cam is not None
        assert cam.label == "Test Cam"

    def test_camera_by_id_missing(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.camera_by_id("nonexistent") is None

    def test_animal_by_id_found(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.animal_by_id("pig1").name == "Piggy"

    def test_animal_by_id_missing(self, config_dir):
        cfg = load_config(config_dir)
        assert cfg.animal_by_id("ghost") is None


# ─── REAL CONFIG FILES SMOKE TEST ─────────────────────────────────────────────

class TestRealConfigFiles:
    def test_real_config_loads(self):
        cfg = load_config("config")
        assert len(cfg.cameras) >= 1
        assert cfg.farm.name == "Bardo Farm"
        assert cfg.gate1.min_brightness > 0
        assert cfg.gate4.min_vibe_score > 0


# ─── VALIDATION FAILURES ───────────────────────────────────────────────────────

class TestConfigValidation:
    def test_missing_cameras_file(self, tmp_path):
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        _write_yaml(tmp_path / "farm.yaml", _minimal_farm())
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path))

    def test_missing_animals_file(self, tmp_path):
        _write_yaml(tmp_path / "cameras.yaml", _minimal_cameras())
        _write_yaml(tmp_path / "farm.yaml", _minimal_farm())
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path))

    def test_missing_farm_file(self, tmp_path):
        _write_yaml(tmp_path / "cameras.yaml", _minimal_cameras())
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path))

    def test_camera_missing_ip(self, tmp_path):
        cameras = {"cameras": [{"id": "cam1", "port": 554, "label": "X"}]}
        _write_yaml(tmp_path / "cameras.yaml", cameras)
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        _write_yaml(tmp_path / "farm.yaml", _minimal_farm())
        with pytest.raises(ConfigError, match="ip"):
            load_config(str(tmp_path))

    def test_empty_cameras_list(self, tmp_path):
        _write_yaml(tmp_path / "cameras.yaml", {"cameras": []})
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        _write_yaml(tmp_path / "farm.yaml", _minimal_farm())
        with pytest.raises(ConfigError, match="at least one camera"):
            load_config(str(tmp_path))

    def test_duplicate_camera_ids(self, tmp_path):
        cameras = {"cameras": [
            {"id": "cam1", "ip": "10.0.0.1", "port": 554, "label": "A"},
            {"id": "cam1", "ip": "10.0.0.2", "port": 554, "label": "B"},
        ]}
        _write_yaml(tmp_path / "cameras.yaml", cameras)
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        _write_yaml(tmp_path / "farm.yaml", _minimal_farm())
        with pytest.raises(ConfigError, match="duplicate"):
            load_config(str(tmp_path))

    def test_farm_missing_voice(self, tmp_path):
        farm = _minimal_farm()
        del farm["farm"]["voice"]
        _write_yaml(tmp_path / "cameras.yaml", _minimal_cameras())
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        _write_yaml(tmp_path / "farm.yaml", farm)
        with pytest.raises(ConfigError, match="voice"):
            load_config(str(tmp_path))

    def test_farm_missing_gate1(self, tmp_path):
        farm = _minimal_farm()
        del farm["gate1"]
        _write_yaml(tmp_path / "cameras.yaml", _minimal_cameras())
        _write_yaml(tmp_path / "animals.yaml", _minimal_animals())
        _write_yaml(tmp_path / "farm.yaml", farm)
        with pytest.raises(ConfigError, match="gate1"):
            load_config(str(tmp_path))

    def test_get_config_before_load_raises(self, monkeypatch):
        import config.loader as loader_module
        monkeypatch.setattr(loader_module, "_config", None)
        with pytest.raises(RuntimeError, match="load_config"):
            get_config()
