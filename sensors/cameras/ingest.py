"""
sensors/cameras/ingest.py
RTSP stream reader for Reolink cameras over Tailscale VPN.

CameraStream wraps a single camera's OpenCV VideoCapture in a background
thread. Frames are decoded and handed to a MotionDetector; when a clip is
ready it is passed to a caller-supplied callback.

CameraManager starts/stops all configured cameras as a group.

Design notes:
  - One thread per camera — they are I/O-bound, not CPU-bound.
  - OpenCV RTSP reconnect: if the stream drops, we retry with exponential
    back-off up to MAX_RECONNECT_ATTEMPTS then log and stop.
  - Credentials are read from CAMERA_USERNAME / CAMERA_PASSWORD env vars.
    Neither value is ever logged.
"""

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional

import cv2

from config.loader import CameraConfig, Gate2Config, MotionConfig
from sensors.cameras.motion import MotionClip, MotionDetector

logger = logging.getLogger(__name__)

# Reconnect policy
_INITIAL_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 60.0
_MAX_RECONNECT_ATTEMPTS = 10


def _build_rtsp_url(cam: CameraConfig) -> str:
    username = os.environ.get("CAMERA_USERNAME", "admin")
    password = os.environ.get("CAMERA_PASSWORD", "")
    return cam.rtsp_url(username, password)


# ─── SINGLE CAMERA STREAM ─────────────────────────────────────────────────────

class CameraStream:
    """
    Manages the RTSP connection and decode loop for one camera.

    Args:
        cam:        CameraConfig for this camera.
        gate2:      Gate2Config — passed to MotionDetector.
        motion_cfg: MotionConfig — passed to MotionDetector.
        clips_dir:  Directory where motion clips are written.
        on_clip:    Called with a MotionClip when an event clip is ready.
    """

    def __init__(
        self,
        cam: CameraConfig,
        gate2: Gate2Config,
        motion_cfg: MotionConfig,
        clips_dir: str,
        on_clip: Callable[[MotionClip], None],
    ) -> None:
        self.cam = cam
        self._on_clip = on_clip

        self._detector = MotionDetector(
            camera_id=cam.id,
            gate2=gate2,
            motion=motion_cfg,
            clips_dir=clips_dir,
        )
        self._detector.on_clip = on_clip

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def camera_id(self) -> str:
        return self.cam.id

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background decode thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"cam-{self.cam.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("CameraStream started: %s (%s)", self.cam.id, self.cam.label)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the decode thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("CameraStream stopped: %s", self.cam.id)

    # ── Decode loop ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = _INITIAL_BACKOFF_SECONDS
        attempts = 0

        while not self._stop_event.is_set():
            cap = self._open_stream()
            if cap is None:
                attempts += 1
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Camera %s: exceeded max reconnect attempts, giving up.",
                        self.cam.id,
                    )
                    self._running = False
                    return
                logger.warning(
                    "Camera %s: connect failed, retry in %.0fs (attempt %d/%d)",
                    self.cam.id, backoff, attempts, _MAX_RECONNECT_ATTEMPTS,
                )
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                continue

            # Successfully connected
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            self._detector.set_fps(fps)
            backoff = _INITIAL_BACKOFF_SECONDS
            attempts = 0
            logger.info("Camera %s: stream open at %.1f fps", self.cam.id, fps)

            self._decode_loop(cap)
            cap.release()

            if not self._stop_event.is_set():
                logger.warning(
                    "Camera %s: stream ended unexpectedly, reconnecting…",
                    self.cam.id,
                )

        self._running = False

    def _open_stream(self) -> Optional[cv2.VideoCapture]:
        url = _build_rtsp_url(self.cam)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Reduce internal buffer to lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _decode_loop(self, cap: cv2.VideoCapture) -> None:
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            try:
                self._detector.process_frame(frame)
            except Exception:
                logger.exception(
                    "Camera %s: error in motion detector", self.cam.id
                )


# ─── MULTI-CAMERA MANAGER ─────────────────────────────────────────────────────

class CameraManager:
    """
    Starts and stops all configured camera streams as a group.

    Usage:
        mgr = CameraManager(config.cameras, config.gate2, config.motion,
                            config.storage.clips_dir, on_clip=queue.put)
        mgr.start_all()
        # ... run until shutdown signal ...
        mgr.stop_all()
    """

    def __init__(
        self,
        cameras: List[CameraConfig],
        gate2: Gate2Config,
        motion_cfg: MotionConfig,
        clips_dir: str,
        on_clip: Callable[[MotionClip], None],
    ) -> None:
        self._streams: Dict[str, CameraStream] = {
            cam.id: CameraStream(
                cam=cam,
                gate2=gate2,
                motion_cfg=motion_cfg,
                clips_dir=clips_dir,
                on_clip=on_clip,
            )
            for cam in cameras
        }

    def start_all(self) -> None:
        for stream in self._streams.values():
            stream.start()
        logger.info("CameraManager: started %d camera(s)", len(self._streams))

    def stop_all(self, timeout: float = 5.0) -> None:
        for stream in self._streams.values():
            stream.stop(timeout=timeout)
        logger.info("CameraManager: all cameras stopped")

    def camera_ids(self) -> List[str]:
        return list(self._streams.keys())

    def stream(self, camera_id: str) -> Optional[CameraStream]:
        return self._streams.get(camera_id)
