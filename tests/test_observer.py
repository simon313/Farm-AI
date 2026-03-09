"""
tests/test_observer.py
Unit tests for the Observer agent.

All Claude API calls are mocked — no real API key needed.
Synthetic video files are written via OpenCV for Gate 1/2 paths.
"""

import json
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from agents.observer.observer import Observer, _parse_json_response, _encode_image
from config.loader import (
    CameraConfig, AnimalConfig, FarmIdentityConfig,
    Gate1Config, Gate2Config, Gate3Config, Gate4Config,
    MotionConfig, StorageConfig, Config,
)
from memory.models import FarmState, Moment
from sensors.cameras.motion import MotionClip


# ─── FIXTURES ────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path) -> Config:
    return Config(
        cameras=[
            CameraConfig(id="north_pen_cam", ip="10.0.0.1", port=554, label="North Pen"),
        ],
        animals=[
            AnimalConfig(
                id="earl", name="Earl", group="north_pen", type="boar",
                markings="large dark grey", temperament="dominant",
                arrival_date="2024-03-15", origin="farm born",
                current_phase="finishing",
            ),
        ],
        farm=FarmIdentityConfig(
            name="Bardo Farm",
            location="Virginia",
            what_we_raise="woods-raised heritage pork",
            voice="honest, earthy",
            what_we_believe="Pigs belong in the woods.",
        ),
        season="spring",
        current_narrative_arc="finishing season",
        products_available=["smoked shoulder"],
        products_coming=[],
        gate1=Gate1Config(
            min_brightness=20, min_sharpness=10.0,
            min_duration_seconds=2, max_shake_threshold=100.0,
            audio_quality_check=False,
        ),
        gate2=Gate2Config(
            min_motion_area_ratio=0.01, min_motion_frames=2,
            background_history=100, background_var_threshold=16,
        ),
        gate3=Gate3Config(min_register_score=0.5, thumbnail_frames=2),
        gate4=Gate4Config(min_vibe_score=0.5),
        motion=MotionConfig(
            sensitivity="medium", clip_pre_buffer_seconds=1,
            clip_post_buffer_seconds=5, max_clip_duration_seconds=30,
        ),
        storage=StorageConfig(
            clips_dir=str(tmp_path / "clips"),
            thumbnails_dir=str(tmp_path / "thumbs"),
            content_dir=str(tmp_path / "content"),
            database_path=str(tmp_path / "test.db"),
            retain_failed_clips=False,
        ),
    )


def _make_farm_state() -> FarmState:
    return FarmState(
        current_season="spring",
        current_narrative_arc="finishing season",
        animals_by_group={"north_pen": ["earl"]},
        products_available=["smoked shoulder"],
        voice="honest, earthy",
        what_we_believe="Pigs belong in the woods.",
    )


def _make_clip(clip_path: str, motion_frame_count: int = 10,
               motion_area_ratio: float = 0.05) -> MotionClip:
    t = time.time()
    return MotionClip(
        camera_id="north_pen_cam",
        clip_path=clip_path,
        start_time=t,
        end_time=t + 8.0,
        duration_seconds=8.0,
        motion_area_ratio=motion_area_ratio,
        motion_frame_count=motion_frame_count,
    )


def _write_good_video(path: str, n_frames: int = 40, fps: float = 10.0) -> str:
    """Write a bright, sharp checkerboard video that passes Gate 1."""
    h, w = 100, 100
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255
        writer.write(frame)
    writer.release()
    return path


def _make_mock_client(gate3_payload: dict, gate4_payload: dict) -> MagicMock:
    """Return an anthropic client mock whose .messages.create returns the given payloads."""
    client = MagicMock()

    def _create(**kwargs):
        # Determine which gate is being called from the prompt content
        prompt_text = ""
        for msg in kwargs.get("messages", []):
            for block in msg.get("content", []):
                if block.get("type") == "text":
                    prompt_text += block["text"]
        if "farm_vibe_score" in prompt_text or "vibe" in prompt_text.lower():
            payload = gate4_payload
        else:
            payload = gate3_payload

        response = MagicMock()
        response.content = [MagicMock(text=json.dumps(payload))]
        return response

    client.messages.create.side_effect = _create
    return client


_GOOD_GATE3 = {
    "warmth": 0.8, "humor": 0.1, "tension": 0.0,
    "reverence": 0.6, "curiosity": 0.3, "information": 0.2,
    "animals_identified": ["Earl"],
    "narrative_note": "Earl roots through mud with focused intensity.",
    "suggested_use": "instagram_reel",
}
_GOOD_GATE4 = {
    "farm_vibe_score": 0.85,
    "vibe_reason": "Unmistakably Bardo Farm — mud, pig, purpose.",
    "conditions": {"light": "overcast", "weather": "clear", "time_of_day": "morning"},
}
_FAIL_GATE3 = {
    "warmth": 0.1, "humor": 0.0, "tension": 0.0,
    "reverence": 0.1, "curiosity": 0.0, "information": 0.0,
    "animals_identified": [],
    "narrative_note": "Empty pen, nothing happening.",
    "suggested_use": "none",
}
_FAIL_GATE4 = {
    "farm_vibe_score": 0.2,
    "vibe_reason": "Generic farm scene with no character.",
    "conditions": {"light": "unclear", "weather": "unclear", "time_of_day": "unclear"},
}


# ─── _parse_json_response ────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_clean_json(self):
        data = {"score": 0.8, "label": "warmth"}
        result = _parse_json_response(json.dumps(data), gate=3)
        assert result == data

    def test_fenced_json(self):
        data = {"score": 0.5}
        text = f"```json\n{json.dumps(data)}\n```"
        result = _parse_json_response(text, gate=3)
        assert result == data

    def test_json_embedded_in_text(self):
        data = {"x": 1}
        text = f"Here is my response: {json.dumps(data)} That's it."
        result = _parse_json_response(text, gate=3)
        assert result == data

    def test_invalid_returns_none(self):
        result = _parse_json_response("not json at all", gate=3)
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_json_response("", gate=3)
        assert result is None


# ─── Observer — happy path ────────────────────────────────────────────────────

class TestObserverHappyPath:
    def test_returns_moment_on_full_pass(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)

        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        client = _make_mock_client(_GOOD_GATE3, _GOOD_GATE4)
        obs = Observer(cfg, _make_farm_state(), client=client)
        moment = obs.evaluate(clip)

        assert moment is not None
        assert isinstance(moment, Moment)

    def test_moment_has_correct_camera_id(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert moment.camera_id == "north_pen_cam"

    def test_moment_location_label_from_config(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert moment.location_label == "North Pen"

    def test_moment_passed_gates_is_true(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert moment.passed_gates is True

    def test_moment_scores_populated(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert moment.technical_score > 0
        assert moment.activity_score > 0
        assert moment.interest_score == pytest.approx(0.8)  # max register from _GOOD_GATE3
        assert moment.farm_vibe_score == pytest.approx(0.85)
        assert 0.0 < moment.overall_score <= 1.0

    def test_moment_narrative_note_from_gate3(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert "Earl" in moment.narrative_note or "mud" in moment.narrative_note

    def test_moment_suggested_use_from_gate3(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert moment.suggested_use == "instagram_reel"

    def test_moment_conditions_from_gate4(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        assert moment.conditions.get("light") == "overcast"
        assert moment.conditions.get("time_of_day") == "morning"

    def test_moment_emotional_register_populated(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(), client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        moment = obs.evaluate(clip)

        # warmth=0.8 and reverence=0.6 both above threshold of 0.5
        assert "warmth" in moment.emotional_register
        assert "reverence" in moment.emotional_register
        assert "humor" not in moment.emotional_register  # 0.1 < 0.5


# ─── Observer — gate failure paths ───────────────────────────────────────────

class TestObserverGateFailures:
    def test_returns_none_when_gate1_fails(self, tmp_path):
        cfg = _make_config(tmp_path)
        # Write a 1-frame video that will be too short for min_duration_seconds=2
        path = str(tmp_path / "short.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10.0, (100, 100))
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        clip = _make_clip(path)
        obs = Observer(cfg, _make_farm_state(),
                       client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        assert obs.evaluate(clip) is None

    def test_returns_none_when_gate2_fails(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        # Zero motion frames — fails Gate 2
        clip = _make_clip(clip_path, motion_frame_count=0, motion_area_ratio=0.0)

        obs = Observer(cfg, _make_farm_state(),
                       client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        assert obs.evaluate(clip) is None

    def test_returns_none_when_gate3_fails(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(),
                       client=_make_mock_client(_FAIL_GATE3, _GOOD_GATE4))
        assert obs.evaluate(clip) is None

    def test_returns_none_when_gate4_fails(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        obs = Observer(cfg, _make_farm_state(),
                       client=_make_mock_client(_GOOD_GATE3, _FAIL_GATE4))
        assert obs.evaluate(clip) is None

    def test_returns_none_when_api_returns_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        # Client raises an exception
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API unavailable")
        obs = Observer(cfg, _make_farm_state(), client=client)
        assert obs.evaluate(clip) is None

    def test_gate1_not_called_on_missing_clip(self, tmp_path):
        cfg = _make_config(tmp_path)
        clip = _make_clip(str(tmp_path / "missing.mp4"))
        obs = Observer(cfg, _make_farm_state(),
                       client=_make_mock_client(_GOOD_GATE3, _GOOD_GATE4))
        result = obs.evaluate(clip)
        assert result is None


# ─── Observer — API call count ────────────────────────────────────────────────

class TestObserverAPICallCount:
    def test_no_api_calls_when_gate1_fails(self, tmp_path):
        cfg = _make_config(tmp_path)
        path = str(tmp_path / "short.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10.0, (100, 100))
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        client = _make_mock_client(_GOOD_GATE3, _GOOD_GATE4)
        obs = Observer(cfg, _make_farm_state(), client=client)
        obs.evaluate(_make_clip(path))

        client.messages.create.assert_not_called()

    def test_no_api_calls_when_gate2_fails(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path, motion_frame_count=0, motion_area_ratio=0.0)

        client = _make_mock_client(_GOOD_GATE3, _GOOD_GATE4)
        obs = Observer(cfg, _make_farm_state(), client=client)
        obs.evaluate(clip)

        client.messages.create.assert_not_called()

    def test_exactly_two_api_calls_on_full_pass(self, tmp_path):
        cfg = _make_config(tmp_path)
        Path(cfg.storage.thumbnails_dir).mkdir(parents=True, exist_ok=True)
        clip_path = _write_good_video(str(tmp_path / "clip.mp4"))
        clip = _make_clip(clip_path)

        client = _make_mock_client(_GOOD_GATE3, _GOOD_GATE4)
        obs = Observer(cfg, _make_farm_state(), client=client)
        obs.evaluate(clip)

        assert client.messages.create.call_count == 2
