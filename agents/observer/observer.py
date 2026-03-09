"""
agents/observer/observer.py
The Observer agent — runs a MotionClip through all four gates and, if it
passes, produces a Moment ready to be saved to the database.

Pipeline per clip:
  Gate 1 (CV)        — technical quality check, no API call
  Gate 2 (CV)        — activity / motion check, no API call
  Gate 3 (Claude)    — emotional register scoring via vision
  Gate 4 (Claude)    — farm vibe scoring via vision

On pass: returns a populated Moment dataclass.
On any gate failure: returns None (caller decides whether to retain clip).

Claude calls use the prompts in agents/observer/prompts/.
Template slots are filled with live FarmState and config values.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic

from agents.observer.gates import (
    GateResult,
    compute_overall_score,
    run_gate1,
    run_gate2,
    run_gate3_stub,
    run_gate4_stub,
)
from config.loader import Config
from memory.models import FarmState, Moment
from sensors.cameras.motion import MotionClip
from sensors.cameras.utils import extract_thumbnails

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_CLAUDE_MODEL = "claude-opus-4-6"


# ─── OBSERVER ────────────────────────────────────────────────────────────────

class Observer:
    """
    Evaluates a MotionClip through the four-gate editorial pipeline.

    Instantiate once at startup; reuse for every clip.

    Args:
        cfg:         Loaded Config (cameras, thresholds, storage paths).
        farm_state:  Current FarmState — season, arc, animals by group.
        client:      Optional anthropic.Anthropic client. If None, one is
                     created from the ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        cfg: Config,
        farm_state: FarmState,
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.cfg = cfg
        self.farm_state = farm_state
        self._client = client or anthropic.Anthropic()
        self._gate3_prompt = _load_prompt("gate3.txt")
        self._gate4_prompt = _load_prompt("gate4.txt")

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, clip: MotionClip) -> Optional[Moment]:
        """
        Run all four gates on clip.

        Returns a Moment if the clip passes all gates, None otherwise.
        Logs the gate result at every stage.
        """
        cam = self.cfg.camera_by_id(clip.camera_id)
        location_label = cam.label if cam else clip.camera_id

        # Gate 1 — technical quality (CV only)
        g1 = run_gate1(clip.clip_path, self.cfg.gate1)
        _log_gate(1, g1, clip)
        if not g1:
            return None

        # Gate 2 — activity (metadata only)
        g2 = run_gate2(clip, self.cfg.gate2)
        _log_gate(2, g2, clip)
        if not g2:
            return None

        # Extract thumbnails for Gates 3 & 4 (shared)
        thumbnails = self._extract_thumbnails(clip)
        if not thumbnails:
            logger.warning("Observer: no thumbnails extracted for %s — skipping AI gates",
                           clip.clip_path)
            return None

        # Gate 3 — emotional register (Claude vision)
        g3_raw = self._call_gate3(clip, thumbnails)
        if g3_raw is None:
            return None
        interest_score = max(
            g3_raw.get("warmth", 0.0),
            g3_raw.get("humor", 0.0),
            g3_raw.get("tension", 0.0),
            g3_raw.get("reverence", 0.0),
            g3_raw.get("curiosity", 0.0),
            g3_raw.get("information", 0.0),
        )
        g3 = run_gate3_stub(interest_score, self.cfg.gate3)
        _log_gate(3, g3, clip)
        if not g3:
            return None

        # Gate 4 — farm vibe (Claude vision)
        g4_raw = self._call_gate4(clip, thumbnails)
        if g4_raw is None:
            return None
        vibe_score = float(g4_raw.get("farm_vibe_score", 0.0))
        g4 = run_gate4_stub(vibe_score, self.cfg.gate4)
        _log_gate(4, g4, clip)
        if not g4:
            return None

        # All gates passed — assemble Moment
        overall = compute_overall_score(g1, g2, g3, g4)

        emotional_register = [
            reg for reg in ("warmth", "humor", "tension", "reverence", "curiosity", "information")
            if g3_raw.get(reg, 0.0) >= self.cfg.gate3.min_register_score
        ]

        conditions = g4_raw.get("conditions", {})

        moment = Moment(
            timestamp=datetime.utcfromtimestamp(clip.start_time),
            duration_seconds=clip.duration_seconds,
            camera_id=clip.camera_id,
            location_label=location_label,
            clip_path=clip.clip_path,
            thumbnail_path=thumbnails[0],
            technical_score=g1.score,
            activity_score=g2.score,
            interest_score=g3.score,
            farm_vibe_score=g4.score,
            overall_score=overall,
            animals_present=list(g3_raw.get("animals_identified", [])),
            emotional_register=emotional_register,
            narrative_note=str(g3_raw.get("narrative_note", ""))[:200],
            suggested_use=str(g3_raw.get("suggested_use", "")),
            conditions=conditions,
            passed_gates=True,
        )

        logger.info(
            "Observer: PASS clip=%s overall=%.2f narrative=%r",
            clip.clip_path, overall, moment.narrative_note,
        )
        return moment

    # ── Gate 3: Claude vision call ─────────────────────────────────────────────

    def _call_gate3(
        self, clip: MotionClip, thumbnails: List[str]
    ) -> Optional[Dict]:
        animals_in_pen = self._animals_for_camera(clip.camera_id)
        prompt = _fill_prompt(
            self._gate3_prompt,
            farm_name=self.cfg.farm.name,
            what_we_raise=self.cfg.farm.what_we_raise,
            voice=self.cfg.farm.voice,
            season=self.farm_state.current_season or self.cfg.season,
            current_narrative_arc=self.farm_state.current_narrative_arc or self.cfg.current_narrative_arc,
            animals_in_pen=animals_in_pen,
            min_register_score=self.cfg.gate3.min_register_score,
        )
        return self._vision_call(prompt, thumbnails, gate=3)

    # ── Gate 4: Claude vision call ─────────────────────────────────────────────

    def _call_gate4(
        self, clip: MotionClip, thumbnails: List[str]
    ) -> Optional[Dict]:
        prompt = _fill_prompt(
            self._gate4_prompt,
            farm_name=self.cfg.farm.name,
            what_we_raise=self.cfg.farm.what_we_raise,
            voice=self.cfg.farm.voice,
            what_we_believe=self.cfg.farm.what_we_believe,
            min_vibe_score=self.cfg.gate4.min_vibe_score,
        )
        return self._vision_call(prompt, thumbnails, gate=4)

    # ── Shared vision helper ───────────────────────────────────────────────────

    def _vision_call(
        self, prompt: str, thumbnail_paths: List[str], gate: int
    ) -> Optional[Dict]:
        """
        Send prompt + images to Claude. Parse and return the JSON response dict.
        Returns None on any API or parse failure.
        """
        content = []
        for path in thumbnail_paths:
            image_data = _encode_image(path)
            if image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                })

        if not content:
            logger.warning("Observer: no valid images for gate %d call", gate)
            return None

        content.append({"type": "text", "text": prompt})

        try:
            response = self._client.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": content}],
            )
            raw_text = response.content[0].text
            return _parse_json_response(raw_text, gate=gate)
        except Exception:
            logger.exception("Observer: API call failed for gate %d", gate)
            return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_thumbnails(self, clip: MotionClip) -> List[str]:
        try:
            return extract_thumbnails(
                clip.clip_path,
                output_dir=self.cfg.storage.thumbnails_dir,
                n=self.cfg.gate3.thumbnail_frames,
            )
        except Exception:
            logger.exception("Observer: thumbnail extraction failed for %s", clip.clip_path)
            return []

    def _animals_for_camera(self, camera_id: str) -> str:
        """
        Build a human-readable list of animals known to be in the pen
        associated with this camera.
        """
        cam = self.cfg.camera_by_id(camera_id)
        if cam is None:
            return "Unknown pen — no animal data available."

        # Match camera id prefix to group name (e.g. north_pen_cam -> north_pen)
        pen_animals = []
        for animal in self.cfg.animals:
            if camera_id.startswith(animal.group) or animal.group in camera_id:
                pen_animals.append(
                    f"- {animal.name} ({animal.type}, {animal.temperament}): {animal.markings}"
                )

        if not pen_animals:
            return f"Pen: {cam.label}. No specific animals registered for this camera."

        return f"Pen: {cam.label}\n" + "\n".join(pen_animals)


# ─── MODULE-LEVEL HELPERS ─────────────────────────────────────────────────────

def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _fill_prompt(template: str, **kwargs) -> str:
    """
    Replace only the named slots {key} that appear in kwargs.
    Leaves all other curly-brace content (e.g. JSON examples in the prompt)
    untouched. This avoids the KeyError that str.format() raises when the
    template contains literal braces that aren't template slots.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _encode_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except OSError:
        logger.warning("Observer: could not read image %s", image_path)
        return None


def _parse_json_response(text: str, gate: int) -> Optional[Dict]:
    """
    Extract and parse a JSON object from Claude's response text.
    Handles cases where Claude wraps the JSON in ```json ... ``` fences.
    Returns None if no valid JSON object is found.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back: find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Observer: could not parse JSON from gate %d response: %r", gate, text[:200])
    return None


def _log_gate(gate_num: int, result: GateResult, clip: MotionClip) -> None:
    level = logging.DEBUG if result.passed else logging.INFO
    logger.log(
        level,
        "Gate %d %s | clip=%s | score=%.2f | %s",
        gate_num,
        "PASS" if result.passed else "FAIL",
        clip.clip_path,
        result.score,
        result.reason,
    )
