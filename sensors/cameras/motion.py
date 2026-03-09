"""
sensors/cameras/motion.py
Motion detection using MOG2 background subtraction and clip extraction.

MotionDetector processes a stream of frames and fires a callback whenever
a motion event is detected and a clip has been saved to disk.

Design:
  - Maintains a rolling ring-buffer (pre_buffer) of recent frames so we
    can prepend footage before the motion trigger.
  - Writes frames to a temporary AVI during the event window, then uses
    ffmpeg (cut_clip) to produce the final MP4 with pre/post buffers.
  - Parameters come directly from config Gate2Config and MotionConfig.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

import cv2
import numpy as np

from config.loader import Gate2Config, MotionConfig


# ─── RESULT DATACLASS ─────────────────────────────────────────────────────────

@dataclass
class MotionClip:
    """Produced by MotionDetector when an event is fully captured."""
    camera_id: str
    clip_path: str                  # final MP4 on disk
    start_time: float               # Unix timestamp of motion trigger
    end_time: float                 # Unix timestamp when motion stopped
    duration_seconds: float
    motion_area_ratio: float        # peak motion area fraction during event
    motion_frame_count: int         # how many frames had qualifying motion


# ─── MOTION DETECTOR ──────────────────────────────────────────────────────────

class MotionDetector:
    """
    Stateful per-camera motion detector.

    Usage:
        detector = MotionDetector(camera_id, gate2_cfg, motion_cfg, clips_dir)
        detector.on_clip = lambda clip: pipeline_queue.put(clip)
        # Feed frames from ingest loop:
        for frame, timestamp in frame_stream:
            detector.process_frame(frame, timestamp)
    """

    def __init__(
        self,
        camera_id: str,
        gate2: Gate2Config,
        motion: MotionConfig,
        clips_dir: str,
    ) -> None:
        self.camera_id = camera_id
        self.gate2 = gate2
        self.motion = motion
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Callback — set by caller before feeding frames
        self.on_clip: Optional[Callable[[MotionClip], None]] = None

        # Background subtractor
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=gate2.background_history,
            varThreshold=gate2.background_var_threshold,
            detectShadows=False,
        )

        # Rolling pre-buffer: (frame, timestamp) tuples
        fps_estimate = 15  # conservative — updated from stream when available
        pre_buf_frames = int(motion.clip_pre_buffer_seconds * fps_estimate)
        self._pre_buffer: Deque[Tuple[np.ndarray, float]] = deque(
            maxlen=max(pre_buf_frames, 1)
        )

        # State
        self._in_event: bool = False
        self._event_frames: List[Tuple[np.ndarray, float]] = []
        self._event_start: float = 0.0
        self._post_buffer_deadline: float = 0.0
        self._motion_frame_count: int = 0
        self._peak_motion_ratio: float = 0.0
        self._fps: float = fps_estimate

        # Writer state (written to temp file during event)
        self._writer: Optional[cv2.VideoWriter] = None
        self._temp_path: Optional[Path] = None

    def set_fps(self, fps: float) -> None:
        """Call with actual stream FPS so pre-buffer sizing is accurate."""
        self._fps = fps
        new_maxlen = max(int(self.motion.clip_pre_buffer_seconds * fps), 1)
        # Rebuild deque preserving existing frames
        old_frames = list(self._pre_buffer)
        self._pre_buffer = deque(old_frames[-new_maxlen:], maxlen=new_maxlen)

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Feed one frame. Returns True if this frame contains qualifying motion.
        Fires self.on_clip callback when a complete event clip is ready.
        """
        if timestamp is None:
            timestamp = time.time()

        h, w = frame.shape[:2]
        frame_area = h * w

        # Background subtraction
        fg_mask = self._bg.apply(frame)
        motion_pixels = int(np.sum(fg_mask > 0))
        motion_ratio = motion_pixels / frame_area if frame_area > 0 else 0.0
        has_motion = motion_ratio >= self.gate2.min_motion_area_ratio

        max_duration = self.motion.max_clip_duration_seconds
        post_buffer = self.motion.clip_post_buffer_seconds

        if not self._in_event:
            self._pre_buffer.append((frame.copy(), timestamp))

            if has_motion:
                # Start event
                self._in_event = True
                self._event_start = timestamp
                self._post_buffer_deadline = timestamp + post_buffer
                self._motion_frame_count = 1
                self._peak_motion_ratio = motion_ratio
                self._event_frames = list(self._pre_buffer)  # include pre-buffer
                self._event_frames.append((frame.copy(), timestamp))
        else:
            self._event_frames.append((frame.copy(), timestamp))

            if has_motion:
                self._motion_frame_count += 1
                self._peak_motion_ratio = max(self._peak_motion_ratio, motion_ratio)
                self._post_buffer_deadline = timestamp + post_buffer

            event_duration = timestamp - self._event_start
            post_expired = timestamp > self._post_buffer_deadline
            hit_max = event_duration >= max_duration

            if post_expired or hit_max:
                self._finalise_event(timestamp)

        return has_motion

    # ── Internal ──────────────────────────────────────────────────────────────

    def _finalise_event(self, end_time: float) -> None:
        """Write accumulated frames to disk and invoke the callback."""
        if not self._event_frames:
            self._reset_event()
            return

        if self._motion_frame_count < self.gate2.min_motion_frames:
            # Too little motion — discard
            self._reset_event()
            return

        clip_path = self._write_clip(self._event_frames)
        if clip_path is None:
            self._reset_event()
            return

        duration = end_time - self._event_start
        clip = MotionClip(
            camera_id=self.camera_id,
            clip_path=str(clip_path),
            start_time=self._event_start,
            end_time=end_time,
            duration_seconds=duration,
            motion_area_ratio=self._peak_motion_ratio,
            motion_frame_count=self._motion_frame_count,
        )

        self._reset_event()

        if self.on_clip:
            self.on_clip(clip)

    def _write_clip(self, frames: List[Tuple[np.ndarray, float]]) -> Optional[Path]:
        """Write frames to an MP4 file. Returns the path or None on failure."""
        if not frames:
            return None

        h, w = frames[0][0].shape[:2]
        ts_str = str(int(self._event_start)).replace(".", "_")
        out_path = self.clips_dir / f"{self.camera_id}_{ts_str}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            return None

        for frame, _ in frames:
            writer.write(frame)
        writer.release()

        return out_path

    def _reset_event(self) -> None:
        self._in_event = False
        self._event_frames = []
        self._event_start = 0.0
        self._post_buffer_deadline = 0.0
        self._motion_frame_count = 0
        self._peak_motion_ratio = 0.0
