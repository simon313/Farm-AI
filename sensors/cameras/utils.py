"""
sensors/cameras/utils.py
Frame-level quality helpers and ffmpeg wrappers used by gates and ingest.

All functions operate on numpy arrays (BGR frames from OpenCV) or file paths.
No disk I/O except where explicitly noted (ffmpeg calls).
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── FRAME QUALITY METRICS ────────────────────────────────────────────────────

def frame_brightness(frame: np.ndarray) -> float:
    """
    Mean pixel brightness of a frame, 0–255.
    Converts to grayscale first so colour channels don't inflate the value.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def frame_sharpness(frame: np.ndarray) -> float:
    """
    Laplacian variance — higher = sharper.
    A blurry frame typically scores below 80; a sharp outdoor scene scores 200+.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_shake(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Estimate camera shake between two consecutive frames.
    Returns the mean absolute pixel-shift magnitude using dense optical flow.
    Higher value = more shake / camera movement.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(magnitude))


def clip_quality_summary(
    frames: List[np.ndarray],
) -> Tuple[float, float, float]:
    """
    Summarise quality of a list of evenly-sampled frames.
    Returns (mean_brightness, mean_sharpness, max_shake).

    max_shake is the highest shake value between any two consecutive frames.
    Returns (0, 0, 0) for empty or single-frame lists.
    """
    if not frames:
        return 0.0, 0.0, 0.0

    brightnesses = [frame_brightness(f) for f in frames]
    sharpnesses = [frame_sharpness(f) for f in frames]
    mean_brightness = float(np.mean(brightnesses))
    mean_sharpness = float(np.mean(sharpnesses))

    if len(frames) < 2:
        return mean_brightness, mean_sharpness, 0.0

    shakes = [frame_shake(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    max_shake = float(max(shakes))

    return mean_brightness, mean_sharpness, max_shake


# ─── FRAME SAMPLING ───────────────────────────────────────────────────────────

def sample_frames(
    video_path: str,
    n: int = 8,
) -> List[np.ndarray]:
    """
    Read n evenly-spaced frames from a video file.
    Returns fewer frames than n if the video is very short.
    Raises RuntimeError if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()
    return frames


def extract_thumbnail(
    video_path: str,
    output_path: str,
    frame_index: int = 0,
) -> str:
    """
    Extract a single frame from a video file and save as JPEG.
    frame_index is 0-based. Returns output_path on success.
    Raises RuntimeError on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read frame {frame_index} from {video_path}"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, frame):
        raise RuntimeError(f"Failed to write thumbnail to {output_path}")

    return output_path


def extract_thumbnails(
    video_path: str,
    output_dir: str,
    n: int = 3,
    base_name: Optional[str] = None,
) -> List[str]:
    """
    Extract n evenly-spaced frames from video_path and save as JPEGs in output_dir.
    Returns list of written file paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= 0:
        return []

    stem = base_name or Path(video_path).stem
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)

    paths = []
    for i, idx in enumerate(indices):
        out = str(Path(output_dir) / f"{stem}_thumb_{i:02d}.jpg")
        extract_thumbnail(video_path, out, frame_index=int(idx))
        paths.append(out)

    return paths


# ─── FFMPEG HELPERS ───────────────────────────────────────────────────────────

def clip_duration_seconds(video_path: str) -> float:
    """
    Return the duration of a video file in seconds using ffprobe.
    Returns 0.0 if the duration cannot be determined.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def cut_clip(
    source_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
) -> str:
    """
    Extract a sub-clip from source_path using ffmpeg (stream copy, no re-encode).
    Returns output_path on success. Raises RuntimeError on ffmpeg failure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-i", source_path,
            "-t", str(duration_seconds),
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg cut_clip failed for {source_path}:\n{result.stderr}"
        )
    return output_path
