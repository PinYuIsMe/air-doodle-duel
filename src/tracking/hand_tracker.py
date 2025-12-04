from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandLandmark:
    """
    Normalized landmark (x, y) in [0, 1].
    """
    x: float
    y: float


class HandTracker:
    """
    Wraps MediaPipe Hands for single-hand tracking.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
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

    def process(self, frame_bgr) -> Optional[List[HandLandmark]]:
        """
        Process a BGR frame and return normalized landmarks of the first hand.

        Returns:
            List[HandLandmark] or None
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]  # first hand
        landmarks = [HandLandmark(lm.x, lm.y)
                     for lm in hand_landmarks.landmark]
        return landmarks

    def get_index_finger_tip(
        self,
        landmarks: List[HandLandmark],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        """
        Convert normalized index finger tip coordinate to pixel coordinate.
        """
        h, w, _ = frame_shape
        index_tip = landmarks[8]  # index finger tip in MediaPipe Hands
        x_px = int(index_tip.x * w)
        y_px = int(index_tip.y * h)
        return x_px, y_px

    def classify_gesture(self, landmarks: List[HandLandmark]) -> str:
        """
        Very simple gesture classifier:
        - "point": fingers extended -> draw
        - "fist": fingers folded -> stop drawing

        NOTE: This is a heuristic. You will likely tune thresholds based on your testing.
        """
        # Use distances between finger tips and wrist in normalized coordinates.
        wrist = landmarks[0]
        finger_tip_indices = [4, 8, 12, 16, 20]

        def dist(a: HandLandmark, b: HandLandmark) -> float:
            dx = a.x - b.x
            dy = a.y - b.y
            return (dx * dx + dy * dy) ** 0.5

        dists = [dist(landmarks[i], wrist) for i in finger_tip_indices]
        avg_dist = sum(dists) / len(dists)

        # Heuristic thresholds; feel free to adjust after trying:
        # - small distance => fist
        # - larger distance => pointing / open hand (we'll treat as "draw")
        if avg_dist < 0.08:
            return "fist"
        else:
            return "point"

    def draw_hand_overlay(self, frame_bgr, landmarks: List[HandLandmark]):
        """
        Optional: draw hand landmarks on the frame for debugging.
        """
        h, w, _ = frame_bgr.shape
        # Reconstruct MediaPipe landmark object to reuse drawing utilities
        hand_landmarks = self.mp_hands.HandLandmark
        # Build dummy object
        from mediapipe.framework.formats import landmark_pb2

        landmark_list = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=0.0) for lm in landmarks
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
