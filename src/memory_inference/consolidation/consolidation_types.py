from __future__ import annotations

from enum import Enum, auto


class UpdateType(Enum):
    NEW = auto()
    REINFORCEMENT = auto()
    SUPERSESSION = auto()
    CONFLICT = auto()
