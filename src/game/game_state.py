from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import time

from .player import Player, PlayerPhase


class Phase(str, Enum):
    COUNTDOWN = "countdown"
    DRAWING = "drawing"
    RESOLVING = "resolving"


@dataclass
class GameState:
    players: Dict[int, Player]
    phase: Phase = Phase.COUNTDOWN
    round_index: int = 1
    countdown_duration: float = 2.0
    drawing_duration: float = 5.0
    resolving_duration: float = 2.0
    phase_end_time: float = field(
        default_factory=lambda: time.perf_counter() + 2.0
    )

    def start_round(self, now: Optional[float] = None):
        if now is None:
            now = time.perf_counter()
        self.phase = Phase.COUNTDOWN
        self.phase_end_time = now + self.countdown_duration
        for p in self.players.values():
            p.state = PlayerPhase.READY

    def update(self, now: Optional[float] = None):
        if now is None:
            now = time.perf_counter()

        if now < self.phase_end_time:
            return

        if self.phase == Phase.COUNTDOWN:
            self.phase = Phase.DRAWING
            self.phase_end_time = now + self.drawing_duration
            for p in self.players.values():
                p.state = PlayerPhase.DRAWING

        elif self.phase == Phase.DRAWING:
            self.phase = Phase.RESOLVING
            self.phase_end_time = now + self.resolving_duration
            for p in self.players.values():
                p.state = PlayerPhase.RESOLVING
            # TODO: Milestone 3 可以在這裡做 doodle 判定、扣血

        elif self.phase == Phase.RESOLVING:
            self.round_index += 1
            self.start_round(now)

    def time_remaining(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.perf_counter()
        return max(0.0, self.phase_end_time - now)
