from typing import Tuple

import cv2

from .game_state import GameState, Phase
from .player import Player


def _draw_hp_bar(frame, player: Player, position: str = "left"):
    h, w, _ = frame.shape
    bar_width = 200
    bar_height = 20
    margin = 20

    if position == "left":
        x0 = margin
    else:
        x0 = w - margin - bar_width
    y0 = margin
    x1 = x0 + bar_width
    y1 = y0 + bar_height

    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)

    hp_ratio = max(0.0, min(1.0, player.hp / player.max_hp))
    inner_w = int(bar_width * hp_ratio)
    cv2.rectangle(
        frame,
        (x0 + 1, y0 + 1),
        (x0 + 1 + inner_w, y1 - 1),
        player.color,
        -1,
    )

    label = f"P{player.player_id} HP: {player.hp}"
    cv2.putText(
        frame,
        label,
        (x0, y0 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_round_info(frame, game_state: GameState, now: float):
    phase_text = f"Round {game_state.round_index} - {game_state.phase.value.upper()}"
    remaining = int(game_state.time_remaining(now)) + 1

    cv2.putText(
        frame,
        phase_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{remaining}s",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if game_state.phase == Phase.RESOLVING:
        text = "RESOLVING..."
        size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        h, w, _ = frame.shape
        x = (w - size[0]) // 2
        y = h - 40
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_split_line(frame):
    h, w, _ = frame.shape
    x = w // 2
    cv2.line(frame, (x, 0), (x, h), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "P1",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "P2",
        (w - 60, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_ui(frame, game_state: GameState, now: float):
    _draw_hp_bar(frame, game_state.players[1], position="left")
    _draw_hp_bar(frame, game_state.players[2], position="right")
    _draw_round_info(frame, game_state, now)
    _draw_split_line(frame)
