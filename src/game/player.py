from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class PlayerPhase(str, Enum):
    IDLE = "idle"
    READY = "ready"
    DRAWING = "drawing"
    RESOLVING = "resolving"


@dataclass
class Player:
    player_id: int
    name: str
    max_hp: int = 100
    hp: int = 100
    state: PlayerPhase = PlayerPhase.IDLE
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR
