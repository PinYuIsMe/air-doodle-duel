from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandLandmark:
    x: float
    y: float


@dataclass
class TrackedHand:
    """
    A single tracked hand with landmarks and handedness label.
    """
    landmarks: List[HandLandmark]
    handedness: str  # "Left" or "Right"


class HandTracker:
    """
    Wraps MediaPipe Hands for up to 2 hands.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

    def process_multi(self, frame_bgr) -> List[TrackedHand]:
        """
        Process a BGR frame and return list of tracked hands.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        tracked_hands: List[TrackedHand] = []
        if not result.multi_hand_landmarks:
            return tracked_hands

        handedness_list = result.multi_handedness or []
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks, handedness_list
        ):
            landmarks = [HandLandmark(lm.x, lm.y)
                         for lm in hand_landmarks.landmark]
            label = handedness.classification[0].label  # "Left" or "Right"
            tracked_hands.append(TrackedHand(
                landmarks=landmarks, handedness=label))

        return tracked_hands

    def get_index_finger_tip(
        self,
        landmarks: List[HandLandmark],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        h, w, _ = frame_shape
        index_tip = landmarks[8]
        x_px = int(index_tip.x * w)
        y_px = int(index_tip.y * h)
        return x_px, y_px

    def classify_gesture(self, landmarks: List[HandLandmark]) -> str:
        """
        - 食指尖 (8) + 中指尖 (12) 接近 => 畫線
        - 兩指分開 => 停止畫線
        """
        def dist(a: HandLandmark, b: HandLandmark) -> float:
            dx = a.x - b.x
            dy = a.y - b.y
            return (dx*dx + dy*dy)**0.5

        index_tip = landmarks[8]   # 食指尖
        middle_tip = landmarks[12]  # 中指尖

        d = dist(index_tip, middle_tip)

        # 這個 threshold 你可以按需要調整（一般 0.03～0.05 效果最好）
        if d < 0.08:
            return "point"   # 畫線
        else:
            return "stop"    # 停止畫線

    def draw_hand_overlay(self, frame_bgr, landmarks: List[HandLandmark]):
        """
        Draw hand landmarks on frame for debugging.
        """
        from mediapipe.framework.formats import landmark_pb2

        landmark_list = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=0.0)
                for lm in landmarks
            ]
        )

        self.mp_drawing.draw_landmarks(
            frame_bgr,
            landmark_list,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_style.get_default_hand_landmarks_style(),
            self.mp_style.get_default_hand_connections_style(),
        )

    def close(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None
