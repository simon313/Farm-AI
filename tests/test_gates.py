"""
tests/test_gates.py
Unit tests for Gate 1, 2, 3, 4 logic with mock/synthetic inputs.

Gate 1 tests use synthetic video files written via OpenCV VideoWriter.
Gate 2 tests use MotionClip dataclasses constructed directly.
Gate 3/4 stub tests use injected float scores.
"""

import time
from pathlib import Path
from typing import List
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from agents.observer.gates import (
    GateResult,
    compute_overall_score,
    run_gate1,
    run_gate2,
    run_gate3_stub,
    run_gate4_stub,
)
from config.loader import Gate1Config, Gate2Config, Gate3Config, Gate4Config
from sensors.cameras.motion import MotionClip


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _make_gate1_cfg(
    min_brightness: int = 30,
    min_sharpness: float = 80.0,
    min_duration_seconds: int = 4,
    max_shake_threshold: float = 15.0,
    audio_quality_check: bool = False,
) -> Gate1Config:
    return Gate1Config(
        min_brightness=min_brightness,
        min_sharpness=min_sharpness,
        min_duration_seconds=min_duration_seconds,
        max_shake_threshold=max_shake_threshold,
        audio_quality_check=audio_quality_check,
    )


def _make_gate2_cfg(
    min_motion_area_ratio: float = 0.02,
    min_motion_frames: int = 5,
) -> Gate2Config:
    return Gate2Config(
        min_motion_area_ratio=min_motion_area_ratio,
        min_motion_frames=min_motion_frames,
        background_history=500,
        background_var_threshold=16,
    )


def _make_clip_meta(
    motion_frame_count: int = 10,
    motion_area_ratio: float = 0.05,
    duration_seconds: float = 8.0,
    camera_id: str = "test_cam",
    clip_path: str = "/tmp/test.mp4",
) -> MotionClip:
    t = time.time()
    return MotionClip(
        camera_id=camera_id,
        clip_path=clip_path,
        start_time=t,
        end_time=t + duration_seconds,
        duration_seconds=duration_seconds,
        motion_area_ratio=motion_area_ratio,
        motion_frame_count=motion_frame_count,
    )


def _write_video(
    path: str,
    n_frames: int = 60,
    fps: float = 10.0,
    value: int = 128,
    size: tuple = (100, 100),
    checkerboard: bool = False,
) -> str:
    """
    Write a synthetic MP4 to path.
    value: solid brightness (0-255) for non-checkerboard frames.
    checkerboard: write high-sharpness checkerboard pattern instead.
    """
    h, w = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(n_frames):
        if checkerboard:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[::2, ::2] = 255
            frame[1::2, 1::2] = 255
        else:
            frame = np.full((h, w, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ─── GATERESULT ───────────────────────────────────────────────────────────────

class TestGateResult:
    def test_bool_true_when_passed(self):
        r = GateResult(passed=True, score=0.9, reason="ok")
        assert bool(r) is True

    def test_bool_false_when_failed(self):
        r = GateResult(passed=False, score=0.1, reason="bad")
        assert bool(r) is False

    def test_details_defaults_to_empty_dict(self):
        r = GateResult(passed=True, score=1.0, reason="x")
        assert r.details == {}


# ─── GATE 1 — TECHNICAL QUALITY ──────────────────────────────────────────────

class TestGate1Duration:
    def test_fails_when_too_short(self, tmp_path):
        path = _write_video(str(tmp_path / "short.mp4"), n_frames=5, fps=10.0)
        cfg = _make_gate1_cfg(min_duration_seconds=10)
        result = run_gate1(path, cfg)
        assert not result.passed
        assert "short" in result.reason.lower() or "duration" in result.reason.lower()

    def test_passes_when_long_enough(self, tmp_path):
        # 60 frames at 10fps = 6s, threshold 4s
        path = _write_video(str(tmp_path / "ok.mp4"), n_frames=60, fps=10.0,
                            value=100, checkerboard=True)
        cfg = _make_gate1_cfg(min_duration_seconds=4, min_brightness=20,
                               min_sharpness=10.0, max_shake_threshold=100.0)
        result = run_gate1(path, cfg)
        assert result.passed, result.reason

    def test_details_contain_duration(self, tmp_path):
        path = _write_video(str(tmp_path / "clip.mp4"), n_frames=60, fps=10.0,
                            value=100, checkerboard=True)
        cfg = _make_gate1_cfg(min_duration_seconds=4, min_brightness=20,
                               min_sharpness=10.0, max_shake_threshold=100.0)
        result = run_gate1(path, cfg)
        assert "duration_seconds" in result.details


class TestGate1Brightness:
    def test_fails_on_dark_clip(self, tmp_path):
        # value=5: very dark
        path = _write_video(str(tmp_path / "dark.mp4"), n_frames=60, fps=10.0, value=5)
        cfg = _make_gate1_cfg(min_brightness=30, min_sharpness=0.0,
                               max_shake_threshold=100.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert not result.passed
        assert "dark" in result.reason.lower() or "brightness" in result.reason.lower()

    def test_passes_on_bright_clip(self, tmp_path):
        path = _write_video(str(tmp_path / "bright.mp4"), n_frames=60, fps=10.0, value=200)
        cfg = _make_gate1_cfg(min_brightness=30, min_sharpness=0.0,
                               max_shake_threshold=100.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert result.passed, result.reason

    def test_details_contain_brightness(self, tmp_path):
        path = _write_video(str(tmp_path / "clip.mp4"), n_frames=60, fps=10.0, value=128)
        cfg = _make_gate1_cfg(min_brightness=30, min_sharpness=0.0,
                               max_shake_threshold=100.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert "mean_brightness" in result.details


class TestGate1Sharpness:
    def test_fails_on_blurry_solid_clip(self, tmp_path):
        # Solid colour = Laplacian variance of 0 = maximally blurry
        path = _write_video(str(tmp_path / "blurry.mp4"), n_frames=60, fps=10.0, value=128)
        cfg = _make_gate1_cfg(min_brightness=20, min_sharpness=500.0,
                               max_shake_threshold=100.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert not result.passed
        assert "blur" in result.reason.lower() or "sharpness" in result.reason.lower()

    def test_passes_on_sharp_checkerboard(self, tmp_path):
        path = _write_video(str(tmp_path / "sharp.mp4"), n_frames=60, fps=10.0,
                            checkerboard=True)
        cfg = _make_gate1_cfg(min_brightness=20, min_sharpness=100.0,
                               max_shake_threshold=100.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert result.passed, result.reason


class TestGate1Shake:
    def test_static_clip_low_shake(self, tmp_path):
        # Identical frames — shake should be ~0
        path = _write_video(str(tmp_path / "static.mp4"), n_frames=60, fps=10.0, value=128)
        cfg = _make_gate1_cfg(min_brightness=20, min_sharpness=0.0,
                               max_shake_threshold=5.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        # May still fail on sharpness, but shake should not be the reason
        if not result.passed:
            assert "shaky" not in result.reason.lower()

    def test_details_contain_shake(self, tmp_path):
        path = _write_video(str(tmp_path / "clip.mp4"), n_frames=60, fps=10.0,
                            checkerboard=True)
        cfg = _make_gate1_cfg(min_brightness=20, min_sharpness=0.0,
                               max_shake_threshold=100.0, min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert "max_shake" in result.details


class TestGate1Score:
    def test_score_between_0_and_1(self, tmp_path):
        path = _write_video(str(tmp_path / "clip.mp4"), n_frames=60, fps=10.0,
                            value=128, checkerboard=True)
        cfg = _make_gate1_cfg(min_duration_seconds=4)
        result = run_gate1(path, cfg)
        assert 0.0 <= result.score <= 1.0

    def test_passing_clip_has_higher_score_than_failing(self, tmp_path):
        good = _write_video(str(tmp_path / "good.mp4"), n_frames=60, fps=10.0,
                            value=200, checkerboard=True)
        bad = _write_video(str(tmp_path / "bad.mp4"), n_frames=5, fps=10.0, value=5)

        cfg_pass = _make_gate1_cfg(min_brightness=20, min_sharpness=10.0,
                                    min_duration_seconds=4, max_shake_threshold=100.0)
        cfg_fail = _make_gate1_cfg(min_brightness=20, min_sharpness=10.0,
                                    min_duration_seconds=10, max_shake_threshold=100.0)
        good_result = run_gate1(good, cfg_pass)
        bad_result = run_gate1(bad, cfg_fail)
        assert good_result.score > bad_result.score

    def test_missing_file_returns_failure(self, tmp_path):
        result = run_gate1(str(tmp_path / "nonexistent.mp4"), _make_gate1_cfg())
        assert not result.passed
        assert result.score == 0.0


# ─── GATE 2 — ACTIVITY ────────────────────────────────────────────────────────

class TestGate2PassingCases:
    def test_passes_with_sufficient_motion(self):
        clip = _make_clip_meta(motion_frame_count=10, motion_area_ratio=0.05)
        cfg = _make_gate2_cfg(min_motion_frames=5, min_motion_area_ratio=0.02)
        result = run_gate2(clip, cfg)
        assert result.passed
        assert result.score > 0

    def test_passes_exactly_at_thresholds(self):
        clip = _make_clip_meta(motion_frame_count=5, motion_area_ratio=0.02)
        cfg = _make_gate2_cfg(min_motion_frames=5, min_motion_area_ratio=0.02)
        result = run_gate2(clip, cfg)
        assert result.passed

    def test_reason_mentions_frame_count_and_ratio(self):
        clip = _make_clip_meta(motion_frame_count=8, motion_area_ratio=0.06)
        cfg = _make_gate2_cfg()
        result = run_gate2(clip, cfg)
        assert "8" in result.reason
        assert "0.06" in result.reason


class TestGate2FailingCases:
    def test_fails_too_few_motion_frames(self):
        clip = _make_clip_meta(motion_frame_count=2, motion_area_ratio=0.10)
        cfg = _make_gate2_cfg(min_motion_frames=5, min_motion_area_ratio=0.02)
        result = run_gate2(clip, cfg)
        assert not result.passed
        assert "frame" in result.reason.lower()

    def test_fails_area_too_small(self):
        clip = _make_clip_meta(motion_frame_count=20, motion_area_ratio=0.001)
        cfg = _make_gate2_cfg(min_motion_frames=5, min_motion_area_ratio=0.02)
        result = run_gate2(clip, cfg)
        assert not result.passed
        assert "area" in result.reason.lower()

    def test_fails_both_metrics(self):
        clip = _make_clip_meta(motion_frame_count=1, motion_area_ratio=0.001)
        cfg = _make_gate2_cfg(min_motion_frames=5, min_motion_area_ratio=0.02)
        result = run_gate2(clip, cfg)
        assert not result.passed
        # Both reasons should appear
        assert "frame" in result.reason.lower()
        assert "area" in result.reason.lower()


class TestGate2Score:
    def test_score_between_0_and_1(self):
        clip = _make_clip_meta(motion_frame_count=10, motion_area_ratio=0.05)
        result = run_gate2(clip, _make_gate2_cfg())
        assert 0.0 <= result.score <= 1.0

    def test_higher_motion_gives_higher_score(self):
        low = _make_clip_meta(motion_frame_count=5, motion_area_ratio=0.02)
        high = _make_clip_meta(motion_frame_count=50, motion_area_ratio=0.30)
        cfg = _make_gate2_cfg()
        assert run_gate2(high, cfg).score > run_gate2(low, cfg).score

    def test_details_contain_expected_keys(self):
        clip = _make_clip_meta()
        result = run_gate2(clip, _make_gate2_cfg())
        for key in ("motion_frame_count", "motion_area_ratio", "duration_seconds"):
            assert key in result.details


# ─── GATE 3 STUB ─────────────────────────────────────────────────────────────

class TestGate3Stub:
    def _cfg(self, threshold: float = 0.6):
        return Gate3Config(min_register_score=threshold, thumbnail_frames=3)

    def test_passes_above_threshold(self):
        result = run_gate3_stub(0.75, self._cfg(0.6))
        assert result.passed
        assert result.score == pytest.approx(0.75)

    def test_fails_below_threshold(self):
        result = run_gate3_stub(0.4, self._cfg(0.6))
        assert not result.passed

    def test_fails_exactly_at_threshold_minus_epsilon(self):
        result = run_gate3_stub(0.599, self._cfg(0.6))
        assert not result.passed

    def test_passes_exactly_at_threshold(self):
        result = run_gate3_stub(0.6, self._cfg(0.6))
        assert result.passed

    def test_reason_contains_score(self):
        result = run_gate3_stub(0.8, self._cfg(0.6))
        assert "0.8" in result.reason or "0.80" in result.reason

    def test_details_contain_interest_score(self):
        result = run_gate3_stub(0.7, self._cfg(0.6))
        assert result.details["interest_score"] == pytest.approx(0.7)


# ─── GATE 4 STUB ─────────────────────────────────────────────────────────────

class TestGate4Stub:
    def _cfg(self, threshold: float = 0.7):
        return Gate4Config(min_vibe_score=threshold)

    def test_passes_above_threshold(self):
        result = run_gate4_stub(0.8, self._cfg(0.7))
        assert result.passed
        assert result.score == pytest.approx(0.8)

    def test_fails_below_threshold(self):
        result = run_gate4_stub(0.5, self._cfg(0.7))
        assert not result.passed

    def test_passes_exactly_at_threshold(self):
        result = run_gate4_stub(0.7, self._cfg(0.7))
        assert result.passed

    def test_reason_contains_score(self):
        result = run_gate4_stub(0.9, self._cfg(0.7))
        assert "0.9" in result.reason or "0.90" in result.reason

    def test_details_contain_vibe_score(self):
        result = run_gate4_stub(0.75, self._cfg(0.7))
        assert result.details["vibe_score"] == pytest.approx(0.75)


# ─── COMPOSITE SCORE ──────────────────────────────────────────────────────────

class TestCompositeScore:
    def _result(self, score: float) -> GateResult:
        return GateResult(passed=True, score=score, reason="ok")

    def test_all_zeros(self):
        assert compute_overall_score(
            self._result(0.0), self._result(0.0),
            self._result(0.0), self._result(0.0),
        ) == pytest.approx(0.0)

    def test_all_ones(self):
        assert compute_overall_score(
            self._result(1.0), self._result(1.0),
            self._result(1.0), self._result(1.0),
        ) == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        from agents.observer.gates import _GATE_WEIGHTS
        assert sum(_GATE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_gate4_has_highest_weight(self):
        from agents.observer.gates import _GATE_WEIGHTS
        assert _GATE_WEIGHTS["gate4"] == max(_GATE_WEIGHTS.values())

    def test_mixed_scores(self):
        # g1=1.0 (w=0.15), g2=0.0 (w=0.20), g3=1.0 (w=0.30), g4=0.0 (w=0.35)
        # expected = 0.15 + 0.0 + 0.30 + 0.0 = 0.45
        score = compute_overall_score(
            self._result(1.0), self._result(0.0),
            self._result(1.0), self._result(0.0),
        )
        assert score == pytest.approx(0.45)
