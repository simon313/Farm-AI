"""
tests/test_sensors.py
Unit tests for the camera sensor layer — utils, motion detection, and ingest.

All tests use synthetic numpy frames; no real camera or video files required.
"""

import time
import threading
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from config.loader import Gate2Config, MotionConfig
from sensors.cameras.utils import (
    clip_quality_summary,
    frame_brightness,
    frame_sharpness,
    frame_shake,
)
from sensors.cameras.motion import MotionDetector, MotionClip


# ─── FRAME FACTORIES ──────────────────────────────────────────────────────────

def _solid_bgr(value: int, h: int = 100, w: int = 100) -> np.ndarray:
    """Return a solid-colour BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _random_frame(h: int = 100, w: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _sharp_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Checkerboard — very high Laplacian variance."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::2, ::2] = 255
    return frame


def _default_gate2() -> Gate2Config:
    return Gate2Config(
        min_motion_area_ratio=0.02,
        min_motion_frames=2,
        background_history=100,
        background_var_threshold=16,
    )


def _default_motion_cfg() -> MotionConfig:
    return MotionConfig(
        sensitivity="medium",
        clip_pre_buffer_seconds=1,
        clip_post_buffer_seconds=1,
        max_clip_duration_seconds=10,
    )


# ─── UTILS: frame_brightness ─────────────────────────────────────────────────

class TestFrameBrightness:
    def test_black_frame_is_zero(self):
        assert frame_brightness(_solid_bgr(0)) == pytest.approx(0.0)

    def test_white_frame_is_255(self):
        assert frame_brightness(_solid_bgr(255)) == pytest.approx(255.0)

    def test_mid_grey(self):
        val = frame_brightness(_solid_bgr(128))
        assert 120 < val < 135

    def test_returns_float(self):
        assert isinstance(frame_brightness(_random_frame()), float)


# ─── UTILS: frame_sharpness ───────────────────────────────────────────────────

class TestFrameSharpness:
    def test_solid_frame_is_zero(self):
        assert frame_sharpness(_solid_bgr(128)) == pytest.approx(0.0)

    def test_sharp_frame_is_high(self):
        assert frame_sharpness(_sharp_frame()) > 1000

    def test_returns_float(self):
        assert isinstance(frame_sharpness(_random_frame()), float)


# ─── UTILS: frame_shake ───────────────────────────────────────────────────────

class TestFrameShake:
    def test_identical_frames_low_shake(self):
        frame = _random_frame(seed=1)
        shake = frame_shake(frame, frame)
        assert shake < 0.5

    def test_shifted_frame_has_higher_shake(self):
        frame_a = _random_frame(seed=42)
        # Shift frame_b by 5 pixels
        frame_b = np.roll(frame_a, shift=5, axis=1)
        shake = frame_shake(frame_a, frame_b)
        assert shake > 0.5

    def test_returns_float(self):
        f = _random_frame(seed=0)
        assert isinstance(frame_shake(f, f), float)


# ─── UTILS: clip_quality_summary ─────────────────────────────────────────────

class TestClipQualitySummary:
    def test_empty_list_returns_zeros(self):
        b, s, k = clip_quality_summary([])
        assert b == 0.0 and s == 0.0 and k == 0.0

    def test_single_frame_shake_is_zero(self):
        _, _, shake = clip_quality_summary([_random_frame()])
        assert shake == 0.0

    def test_brightness_aggregated(self):
        frames = [_solid_bgr(50), _solid_bgr(150)]
        brightness, _, _ = clip_quality_summary(frames)
        assert brightness == pytest.approx(100.0, abs=5)

    def test_solid_frames_sharpness_zero(self):
        frames = [_solid_bgr(100), _solid_bgr(100)]
        _, sharpness, _ = clip_quality_summary(frames)
        assert sharpness == pytest.approx(0.0)

    def test_multi_frame_shake_non_negative(self):
        frames = [_random_frame(seed=i) for i in range(5)]
        _, _, shake = clip_quality_summary(frames)
        assert shake >= 0.0


# ─── MOTION DETECTOR ─────────────────────────────────────────────────────────

class TestMotionDetectorNoMotion:
    """Feed static background frames — no clip should fire."""

    def test_no_clip_on_static_scene(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append

        bg = _solid_bgr(100)
        t = time.time()
        for i in range(30):
            det.process_frame(bg, timestamp=t + i * 0.1)

        assert len(clips) == 0


class TestMotionDetectorMotionEvent:
    """Feed background then a very different foreground — clip should fire."""

    def _feed_motion_event(self, det: MotionDetector, tmp_path, bg_count=30) -> None:
        t = time.time()
        bg = _solid_bgr(100, h=200, w=200)

        # Warm up background model
        for i in range(bg_count):
            det.process_frame(bg.copy(), timestamp=t + i * 0.1)

        # Inject motion frames: bright pixels on dark background
        motion_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        motion_frame[50:150, 50:150] = 255   # large white rectangle = clear foreground

        t_motion = t + bg_count * 0.1
        for j in range(20):
            det.process_frame(motion_frame.copy(), timestamp=t_motion + j * 0.1)

        # Let post-buffer expire
        post_end = t_motion + 20 * 0.1 + det.motion.clip_post_buffer_seconds + 0.5
        det.process_frame(bg.copy(), timestamp=post_end)

    def test_clip_fires_after_event(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append
        self._feed_motion_event(det, tmp_path)
        assert len(clips) == 1

    def test_clip_has_correct_camera_id(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("north_pen_cam", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append
        self._feed_motion_event(det, tmp_path)
        assert clips[0].camera_id == "north_pen_cam"

    def test_clip_file_written_to_disk(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append
        self._feed_motion_event(det, tmp_path)
        assert len(clips) == 1
        import os
        assert os.path.exists(clips[0].clip_path)

    def test_clip_has_positive_duration(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append
        self._feed_motion_event(det, tmp_path)
        assert clips[0].duration_seconds > 0

    def test_clip_motion_frame_count_positive(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append
        self._feed_motion_event(det, tmp_path)
        assert clips[0].motion_frame_count >= _default_gate2().min_motion_frames

    def test_clip_motion_area_ratio_positive(self, tmp_path):
        clips: List[MotionClip] = []
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append
        self._feed_motion_event(det, tmp_path)
        assert clips[0].motion_area_ratio > 0


class TestMotionDetectorMinMotionFrames:
    """Events with too few motion frames should be silently discarded."""

    def test_brief_flicker_discarded(self, tmp_path):
        clips: List[MotionClip] = []
        gate2 = Gate2Config(
            min_motion_area_ratio=0.02,
            min_motion_frames=5,        # require 5 motion frames
            background_history=100,
            background_var_threshold=16,
        )
        det = MotionDetector("cam1", gate2, _default_motion_cfg(), str(tmp_path))
        det.on_clip = clips.append

        t = time.time()
        bg = _solid_bgr(100, h=200, w=200)

        # Warm up
        for i in range(30):
            det.process_frame(bg.copy(), timestamp=t + i * 0.1)

        # Just 2 motion frames (below min_motion_frames=5)
        motion_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        motion_frame[50:150, 50:150] = 255
        t_m = t + 3.0
        det.process_frame(motion_frame.copy(), timestamp=t_m)
        det.process_frame(motion_frame.copy(), timestamp=t_m + 0.1)

        # Expire post-buffer
        det.process_frame(bg.copy(), timestamp=t_m + 5.0)

        assert len(clips) == 0


class TestMotionDetectorMaxDuration:
    """Event should be finalised when max_clip_duration is hit."""

    def test_event_capped_at_max_duration(self, tmp_path):
        clips: List[MotionClip] = []
        motion_cfg = MotionConfig(
            sensitivity="medium",
            clip_pre_buffer_seconds=1,
            clip_post_buffer_seconds=30,    # long post-buffer
            max_clip_duration_seconds=3,    # but hard cap at 3s
        )
        det = MotionDetector("cam1", _default_gate2(), motion_cfg, str(tmp_path))
        det.on_clip = clips.append

        t = time.time()
        bg = _solid_bgr(100, h=200, w=200)
        for i in range(30):
            det.process_frame(bg.copy(), timestamp=t + i * 0.1)

        motion_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        motion_frame[50:150, 50:150] = 255
        t_m = t + 3.0

        # Feed motion for 4 seconds — should be capped at 3
        for j in range(40):
            det.process_frame(motion_frame.copy(), timestamp=t_m + j * 0.1)

        assert len(clips) >= 1
        assert clips[0].duration_seconds <= motion_cfg.max_clip_duration_seconds + 0.5


class TestMotionDetectorSetFps:
    """set_fps() should resize the pre-buffer without crashing."""

    def test_set_fps_preserves_frames(self, tmp_path):
        det = MotionDetector("cam1", _default_gate2(), _default_motion_cfg(), str(tmp_path))
        t = time.time()
        bg = _solid_bgr(100)
        for i in range(5):
            det.process_frame(bg.copy(), timestamp=t + i * 0.1)
        det.set_fps(30.0)
        assert det._pre_buffer.maxlen is not None
        assert det._pre_buffer.maxlen > 0


# ─── INGEST: CameraStream unit tests ─────────────────────────────────────────

class TestCameraStreamBasic:
    """
    CameraStream is tested with a mocked VideoCapture — no real RTSP stream.
    """

    def _make_stream(self, tmp_path, clips=None):
        from config.loader import CameraConfig
        from sensors.cameras.ingest import CameraStream

        cam = CameraConfig(id="test_cam", ip="10.0.0.1", port=554, label="Test")
        on_clip = (clips.append if clips is not None else lambda c: None)
        return CameraStream(
            cam=cam,
            gate2=_default_gate2(),
            motion_cfg=_default_motion_cfg(),
            clips_dir=str(tmp_path),
            on_clip=on_clip,
        )

    def test_is_not_running_before_start(self, tmp_path):
        stream = self._make_stream(tmp_path)
        assert not stream.is_running

    def test_camera_id_matches_config(self, tmp_path):
        stream = self._make_stream(tmp_path)
        assert stream.camera_id == "test_cam"

    def test_stop_before_start_is_safe(self, tmp_path):
        stream = self._make_stream(tmp_path)
        stream.stop()  # should not raise

    def test_start_sets_running(self, tmp_path):
        """Start with a VideoCapture that immediately returns no frames."""
        stream = self._make_stream(tmp_path)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 15.0
        mock_cap.read.return_value = (False, None)  # stream ends immediately

        with patch("sensors.cameras.ingest.cv2.VideoCapture", return_value=mock_cap):
            stream.start()
            time.sleep(0.2)
            stream.stop()

        # After stop() completes, running should be False
        assert not stream.is_running


class TestCameraManagerBasic:
    def test_camera_ids_match_config(self, tmp_path):
        from config.loader import CameraConfig
        from sensors.cameras.ingest import CameraManager

        cams = [
            CameraConfig(id="cam_a", ip="10.0.0.1", port=554, label="A"),
            CameraConfig(id="cam_b", ip="10.0.0.2", port=554, label="B"),
        ]
        mgr = CameraManager(
            cameras=cams,
            gate2=_default_gate2(),
            motion_cfg=_default_motion_cfg(),
            clips_dir=str(tmp_path),
            on_clip=lambda c: None,
        )
        assert set(mgr.camera_ids()) == {"cam_a", "cam_b"}

    def test_stream_lookup_by_id(self, tmp_path):
        from config.loader import CameraConfig
        from sensors.cameras.ingest import CameraManager

        cams = [CameraConfig(id="cam_x", ip="10.0.0.3", port=554, label="X")]
        mgr = CameraManager(
            cameras=cams,
            gate2=_default_gate2(),
            motion_cfg=_default_motion_cfg(),
            clips_dir=str(tmp_path),
            on_clip=lambda c: None,
        )
        assert mgr.stream("cam_x") is not None
        assert mgr.stream("nonexistent") is None
