from typing import List, Optional, Tuple

import cv2
import numpy as np


class CanvasManager:
    """
    Manages the drawing canvas and stroke history.
    """

    def __init__(
        self,
        width: int,
        height: int,
        stroke_color: Tuple[int, int, int] = (0, 255, 0),
        stroke_thickness: int = 4,
    ):
        self.width = width
        self.height = height
        self.stroke_color = stroke_color
        self.stroke_thickness = stroke_thickness

        # Separate transparent-like canvas to draw strokes
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # For continuous lines
        self.prev_point: Optional[Tuple[int, int]] = None

    def reset(self):
        self.canvas[:] = 0
        self.prev_point = None

    def update_drawing(self, point: Tuple[int, int], is_drawing: bool):
        """
        Update the canvas with a new point if drawing is active.

        Args:
            point: (x, y) in pixel coordinates
            is_drawing: whether to draw or not
        """
        if not is_drawing:
            # When not drawing, we reset the previous point so we don't connect lines.
            self.prev_point = None
            return

        if self.prev_point is None:
            # Only set starting point; wait for next frame to draw a line.
            self.prev_point = point
            return

        cv2.line(
            self.canvas,
            self.prev_point,
            point,
            self.stroke_color,
            self.stroke_thickness,
        )
        self.prev_point = point

    def overlay_on_frame(self, frame_bgr):
        """
        Alpha-blend the drawing canvas on top of the original camera frame.
        如果 canvas 大小跟 frame 不同，就先 resize 一下避免 OpenCV error。
        """
        fh, fw, _ = frame_bgr.shape
        ch, cw, _ = self.canvas.shape

        if (fh, fw) != (ch, cw):
            # 這種情況通常是相機解析度被調整，或初始化假設跟真實不同
            self.canvas = cv2.resize(self.canvas, (fw, fh))

        alpha = 0.3
        cv2.addWeighted(self.canvas, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
        return frame_bgr
