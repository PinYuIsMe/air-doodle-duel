import time

import cv2

from tracking.camera import Camera
from tracking.hand_tracker import HandTracker
from drawing.canvas_manager import CanvasManager


def main():
    # Camera and drawing setup
    width, height = 640, 480
    camera = Camera(device_index=0, width=width, height=height)
    hand_tracker = HandTracker(max_num_hands=1)
    canvas_manager = CanvasManager(width=width, height=height)

    window_name = "Air-Doodle Duel - Single Player Drawing"

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            # Flip horizontally so it feels like a mirror
            frame = cv2.flip(frame, 1)

            # 1) Hand tracking
            landmarks = hand_tracker.process(frame)

            is_drawing = False
            index_tip_point = None

            if landmarks is not None:
                # Simple gesture: "point" => draw, "fist" => stop
                gesture = hand_tracker.classify_gesture(landmarks)
                if gesture == "point":
                    is_drawing = True

                # Get fingertip coordinates
                index_tip_point = hand_tracker.get_index_finger_tip(
                    landmarks, frame.shape
                )

                # Optional: draw hand landmarks overlay for debugging
                hand_tracker.draw_hand_overlay(frame, landmarks)

            # 2) Update drawing canvas
            if index_tip_point is not None:
                canvas_manager.update_drawing(index_tip_point, is_drawing)
            else:
                # No hand detected => stop drawing
                canvas_manager.update_drawing((0, 0), False)

            # 3) Overlay strokes on camera frame
            frame = canvas_manager.overlay_on_frame(frame)

            # 4) Compute FPS
            current_time = time.perf_counter()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = current_time

            # 5) Draw status text
            status_text = f"FPS: {fps:.1f} | Drawing: {'ON' if is_drawing else 'OFF'}"
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'c' to clear, 'q' or ESC to quit.",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c"):
                canvas_manager.reset()

    finally:
        camera.release()
        hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
