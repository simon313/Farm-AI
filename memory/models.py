"""
memory/models.py
All data model definitions for the Bardo Farm AI System.
Python dataclasses are the canonical data model. Serialization to/from SQLite
happens only in memory/db.py — never in this file.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional
import uuid


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Moment:
    """
    The atomic unit of the system. Every interesting thing the farm produces.
    Created by the observer agent after a clip passes all four gates.
    """
    # Identity
    id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    camera_id: str = ""                   # references a key in config/cameras.yaml
    location_label: str = ""              # human name e.g. 'north pen', 'oak grove trail'
    clip_path: str = ""                   # absolute path to video file on Dell
    thumbnail_path: str = ""              # frame grab used for AI calls and dashboard
    audio_path: Optional[str] = None      # extracted audio if available

    # Gate scores — populated by observer agent
    technical_score: float = 0.0          # Gate 1: 0.0 to 1.0
    activity_score: float = 0.0           # Gate 2: 0.0 to 1.0
    interest_score: float = 0.0           # Gate 3: AI judgment — 0.0 to 1.0
    farm_vibe_score: float = 0.0          # Gate 4: AI judgment — 0.0 to 1.0
    overall_score: float = 0.0            # weighted composite of all gate scores

    # Editorial metadata — populated by observer agent
    tags: List[str] = field(default_factory=list)
    animals_present: List[str] = field(default_factory=list)    # animal IDs from animals.yaml
    emotional_register: List[str] = field(default_factory=list)
    # Valid values: 'warmth', 'humor', 'tension', 'reverence', 'curiosity', 'information'
    narrative_note: str = ""              # ONE sentence max — observer's editorial judgment
    suggested_use: str = ""              # observer's content type recommendation
    conditions: Dict[str, str] = field(default_factory=dict)
    # e.g. {'light': 'golden hour', 'weather': 'overcast', 'time_of_day': 'morning'}

    # Status flags
    passed_gates: bool = False
    reviewed: bool = False                # human has seen it on dashboard
    used_in_content: List[str] = field(default_factory=list)    # content IDs
    archived: bool = False


@dataclass
class Animal:
    """
    Individual pig — a character the audience follows over time.
    Defined initially in config/animals.yaml; extended here with event history.
    """
    id: str = field(default_factory=_new_id)
    name: str = ""                        # e.g. 'Earl', 'The Sisters'
    group: str = ""                       # pen name or 'woods-roaming'
    type: str = ""                        # 'boar', 'sow', 'piglet', 'finisher'
    markings: str = ""                    # physical description for visual ID by observer
    temperament: str = ""                 # 'curious', 'shy', 'dominant', 'social'
    arrival_date: Optional[date] = None
    origin: str = ""                      # where the animal came from
    current_phase: str = ""               # 'growing', 'finishing', 'harvested'
    harvest_date: Optional[date] = None
    moment_appearances: List[str] = field(default_factory=list)   # moment IDs
    milestones: List[Dict] = field(default_factory=list)
    # e.g. [{'date': '2025-03-01', 'description': 'moved to finishing pen'}]


@dataclass
class Content:
    """
    Assembled content created by the producer and storyteller agents from one
    or more Moments. Moves through draft -> approved -> posted lifecycle.
    """
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=datetime.utcnow)
    type: str = ""                        # 'post', 'reel', 'story', 'email', 'dispatch'
    format: str = ""                      # 'video', 'image', 'text', 'carousel'
    source_moment_ids: List[str] = field(default_factory=list)
    source_animal_ids: List[str] = field(default_factory=list)
    media_path: str = ""                  # path to assembled media file
    caption: str = ""
    hook: str = ""                        # first line — most important element
    hashtags: List[str] = field(default_factory=list)
    emotional_tone: str = ""              # 'warm', 'honest', 'humorous', 'reverent', 'informative'
    farm_narrative_arc: str = ""          # where in the larger story this content sits
    cta_included: bool = False
    product_referenced: Optional[str] = None
    link: Optional[str] = None
    platforms: List[str] = field(default_factory=list)    # ['instagram', 'tiktok', ...]
    scheduled_for: Optional[datetime] = None
    status: str = "draft"                 # 'draft', 'approved', 'posted'
    performance: Dict = field(default_factory=dict)
    # e.g. {'instagram': {'likes': 0, 'shares': 0, 'comments': 0}}


@dataclass
class Person:
    """
    Everyone the farm has a relationship with — customers, followers, prospects, press.
    Managed by the relationship agent.
    """
    id: str = field(default_factory=_new_id)
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None        # rough — state or region
    type: str = ""                        # 'customer', 'follower', 'prospect', 'press'
    first_contact: Optional[date] = None
    channel: str = ""                     # how they found the farm
    warmth: float = 0.0                   # 0.0 to 1.0 — relationship strength score
    purchases: List[Dict] = field(default_factory=list)
    # e.g. [{'date': '2025-02-14', 'product': 'smoked shoulder', 'amount': 45.00}]
    content_engaged: List[str] = field(default_factory=list)    # content IDs
    conversations: List[Dict] = field(default_factory=list)     # relationship agent interaction log
    visited_farm: bool = False
    preferred_products: List[str] = field(default_factory=list)
    communication_cadence: str = "monthly"    # 'weekly', 'monthly', 'event-driven'
    next_action_type: str = "none"            # 'follow_up', 'offer', 'share_moment', 'none'
    next_action_reason: str = ""              # why now
    next_action_content: str = ""             # what to send them


@dataclass
class FarmState:
    """
    Singleton. The system's shared understanding of the farm right now.
    Every agent reads FarmState before acting.
    Loaded from config/farm.yaml + live data from the database at startup.
    """
    current_season: str = ""
    current_narrative_arc: str = ""       # e.g. 'new piglets arriving', 'finishing season'
    animals_by_group: Dict[str, List[str]] = field(default_factory=dict)
    # e.g. {'north_pen': ['earl'], 'oak_grove': ['the_sisters']}
    products_available: List[str] = field(default_factory=list)
    products_coming: List[Dict] = field(default_factory=list)
    # e.g. [{'product': 'summer sausage', 'estimated_date': '2025-07-01'}]
    recent_weather: List[Dict] = field(default_factory=list)    # last 7 days
    current_conditions: str = ""          # right now on the farm
    notable_events: List[Dict] = field(default_factory=list)
    # e.g. [{'date': '2025-03-01', 'description': 'first piglets of season born'}]
    voice: str = ""                       # tone descriptor e.g. 'honest, earthy, direct'
    what_we_believe: str = ""             # farm philosophy in one paragraph
