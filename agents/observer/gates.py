"""
agents/observer/gates.py
Four-gate editorial filter. Each gate returns a GateResult.

Gate 1 — Technical quality    (pure CV, no AI call)
Gate 2 — Activity / motion    (derived from MotionClip metadata + CV)
Gate 3 — Emotional register   (Claude vision, async)
Gate 4 — Farm vibe            (Claude vision, async)

Gates 1 and 2 are implemented here and tested fully offline.
Gates 3 and 4 stubs are included; their Claude calls are filled in by
agents/observer/observer.py which handles prompt loading and API wiring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

from config.loader import Gate1Config, Gate2Config
from sensors.cameras.motion import MotionClip
from sensors.cameras.utils import clip_quality_summary, sample_frames

logger = logging.getLogger(__name__)


# ─── SHARED RESULT TYPE ───────────────────────────────────────────────────────

@dataclass
class GateResult:
    """
    Outcome of a single gate evaluation.

    passed:      Whether the clip clears this gate.
    score:       Normalised 0.0-1.0 score for this gate.
    reason:      Human-readable explanation (always populated, pass or fail).
    details:     Optional per-metric breakdown for debugging / dashboard.
    """
    passed: bool
    score: float
    reason: str
    details: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


# ─── GATE 1 — TECHNICAL QUALITY ──────────────────────────────────────────────

def run_gate1(clip_path: str, cfg: Gate1Config, n_sample_frames: int = 8) -> GateResult:
    """
    Pure computer-vision technical quality check. No AI call.

    Checks (all must pass):
      1. Clip duration >= cfg.min_duration_seconds
      2. Mean frame brightness >= cfg.min_brightness
      3. Mean frame sharpness (Laplacian variance) >= cfg.min_sharpness
      4. Peak inter-frame shake <= cfg.max_shake_threshold

    Score is the mean of the four sub-scores, each normalised 0-1.
    """
    # Duration — derived from OpenCV so no ffprobe dependency is required
    _cap = cv2.VideoCapture(clip_path)
    if not _cap.isOpened():
        return GateResult(
            passed=False,
            score=0.0,
            reason=f"Could not open clip: {clip_path}",
        )
    _fps = _cap.get(cv2.CAP_PROP_FPS) or 25.0
    _frames = _cap.get(cv2.CAP_PROP_FRAME_COUNT)
    _cap.release()
    duration = _frames / _fps if _fps > 0 else 0.0

    if duration < cfg.min_duration_seconds:
        return GateResult(
            passed=False,
            score=0.0,
            reason=f"Too short: {duration:.1f}s < {cfg.min_duration_seconds}s minimum",
            details={"duration_seconds": duration},
        )

    # Sample frames
    try:
        frames = sample_frames(clip_path, n=n_sample_frames)
    except RuntimeError as exc:
        return GateResult(
            passed=False,
            score=0.0,
            reason=f"Could not read frames: {exc}",
        )

    if not frames:
        return GateResult(
            passed=False,
            score=0.0,
            reason="No frames decoded from clip",
        )

    brightness, sharpness, max_shake = clip_quality_summary(frames)

    # Per-metric pass/fail
    bright_ok = brightness >= cfg.min_brightness
    sharp_ok = sharpness >= cfg.min_sharpness
    shake_ok = max_shake <= cfg.max_shake_threshold

    failures = []
    if not bright_ok:
        failures.append(
            f"too dark (brightness {brightness:.1f} < {cfg.min_brightness})"
        )
    if not sharp_ok:
        failures.append(
            f"too blurry (sharpness {sharpness:.1f} < {cfg.min_sharpness})"
        )
    if not shake_ok:
        failures.append(
            f"too shaky (shake {max_shake:.2f} > {cfg.max_shake_threshold})"
        )

    passed = not failures

    # Normalised sub-scores (capped at 1.0)
    brightness_score = min(brightness / max(cfg.min_brightness, 1), 2.0) / 2.0
    sharpness_score = min(sharpness / max(cfg.min_sharpness, 1), 2.0) / 2.0
    # Shake: lower is better — invert so 0 shake = 1.0 score
    shake_score = max(0.0, 1.0 - max_shake / max(cfg.max_shake_threshold, 1))
    duration_score = min(duration / max(cfg.min_duration_seconds, 1), 2.0) / 2.0

    score = (brightness_score + sharpness_score + shake_score + duration_score) / 4.0
    score = min(max(score, 0.0), 1.0)

    if passed:
        reason = (
            f"Technical quality OK — brightness {brightness:.1f}, "
            f"sharpness {sharpness:.1f}, shake {max_shake:.2f}, "
            f"duration {duration:.1f}s"
        )
    else:
        reason = "Technical quality failed: " + "; ".join(failures)

    return GateResult(
        passed=passed,
        score=score,
        reason=reason,
        details={
            "duration_seconds": duration,
            "mean_brightness": brightness,
            "mean_sharpness": sharpness,
            "max_shake": max_shake,
            "brightness_score": brightness_score,
            "sharpness_score": sharpness_score,
            "shake_score": shake_score,
            "duration_score": duration_score,
        },
    )


# ─── GATE 2 — ACTIVITY ────────────────────────────────────────────────────────

def run_gate2(clip: MotionClip, cfg: Gate2Config) -> GateResult:
    """
    Activity gate — uses metadata already captured by MotionDetector.
    No additional CV processing needed; all values were recorded during ingest.

    Checks (all must pass):
      1. motion_frame_count >= cfg.min_motion_frames
      2. motion_area_ratio  >= cfg.min_motion_area_ratio

    Score: mean of the two sub-scores, capped 0-1.
    """
    frame_ok = clip.motion_frame_count >= cfg.min_motion_frames
    area_ok = clip.motion_area_ratio >= cfg.min_motion_area_ratio

    failures = []
    if not frame_ok:
        failures.append(
            f"too few motion frames ({clip.motion_frame_count} < {cfg.min_motion_frames})"
        )
    if not area_ok:
        failures.append(
            f"motion area too small ({clip.motion_area_ratio:.3f} < {cfg.min_motion_area_ratio})"
        )

    passed = not failures

    # Normalised scores
    frame_score = min(clip.motion_frame_count / max(cfg.min_motion_frames, 1), 2.0) / 2.0
    area_score = min(clip.motion_area_ratio / max(cfg.min_motion_area_ratio, 0.001), 2.0) / 2.0
    score = min(max((frame_score + area_score) / 2.0, 0.0), 1.0)

    if passed:
        reason = (
            f"Activity OK — {clip.motion_frame_count} motion frames, "
            f"area ratio {clip.motion_area_ratio:.3f}"
        )
    else:
        reason = "Activity check failed: " + "; ".join(failures)

    return GateResult(
        passed=passed,
        score=score,
        reason=reason,
        details={
            "motion_frame_count": clip.motion_frame_count,
            "motion_area_ratio": clip.motion_area_ratio,
            "duration_seconds": clip.duration_seconds,
            "frame_score": frame_score,
            "area_score": area_score,
        },
    )


# ─── GATE 3 — EMOTIONAL REGISTER (stub) ──────────────────────────────────────

def run_gate3_stub(interest_score: float, cfg) -> GateResult:
    """
    Stub for Gate 3. The real implementation in observer.py calls Claude vision
    and passes the parsed score here for final pass/fail evaluation.
    """
    passed = interest_score >= cfg.min_register_score
    return GateResult(
        passed=passed,
        score=interest_score,
        reason=(
            f"Emotional register score {interest_score:.2f} "
            f"{'passes' if passed else 'fails'} threshold {cfg.min_register_score}"
        ),
        details={"interest_score": interest_score},
    )


# ─── GATE 4 — FARM VIBE (stub) ────────────────────────────────────────────────

def run_gate4_stub(vibe_score: float, cfg) -> GateResult:
    """
    Stub for Gate 4. The real implementation in observer.py calls Claude vision
    and passes the parsed score here for final pass/fail evaluation.
    """
    passed = vibe_score >= cfg.min_vibe_score
    return GateResult(
        passed=passed,
        score=vibe_score,
        reason=(
            f"Farm vibe score {vibe_score:.2f} "
            f"{'passes' if passed else 'fails'} threshold {cfg.min_vibe_score}"
        ),
        details={"vibe_score": vibe_score},
    )


# ─── COMPOSITE SCORE ──────────────────────────────────────────────────────────

# Weights reflect editorial priority: vibe and register matter most
_GATE_WEIGHTS = {
    "gate1": 0.15,
    "gate2": 0.20,
    "gate3": 0.30,
    "gate4": 0.35,
}


def compute_overall_score(
    g1: GateResult,
    g2: GateResult,
    g3: GateResult,
    g4: GateResult,
) -> float:
    """Weighted composite of all four gate scores. Returns 0.0-1.0."""
    return (
        g1.score * _GATE_WEIGHTS["gate1"]
        + g2.score * _GATE_WEIGHTS["gate2"]
        + g3.score * _GATE_WEIGHTS["gate3"]
        + g4.score * _GATE_WEIGHTS["gate4"]
    )
