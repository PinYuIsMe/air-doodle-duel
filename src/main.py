import time

import cv2

from tracking.camera import Camera
from tracking.hand_tracker import HandTracker, TrackedHand
from drawing.canvas_manager import CanvasManager
from game.player import Player
from game.game_state import GameState, Phase
from game.ui_overlay import draw_ui


def point_in_box(point, box):
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def clamp_point_to_box(point, box):
    x, y = point
    x0, y0, x1, y1 = box
    x = min(max(x, x0), x1)
    y = min(max(y, y0), y1)
    return (x, y)


def draw_fingertip_indicator(frame, point, is_drawing, color):
    """
    在指尖畫一個亮圈：
    - is_drawing == True 時用亮色實心 + 外框
    - is_drawing == False 時用淡色小點（可選）
    """
    if point is None:
        return

    x, y = point

    if is_drawing:
        # 發亮狀態：實心亮圈 + 外環
        cv2.circle(frame, (x, y), 10, color, -1)     # 實心
        cv2.circle(frame, (x, y), 16, (0, 255, 255), 2)  # 外環（亮黃）
    else:
        # 可選：非畫線狀態畫一個淡淡的小點（你不想要可以拿掉）
        cv2.circle(frame, (x, y), 6, (180, 180, 180), 1)


def main():

    # 不要先假設寬高，讓 Camera 優先決定
    camera = Camera(device_index=0)

    # 先讀一幀，拿到實際 frame 尺寸
    success, frame = camera.read()
    if not success:
        raise RuntimeError("Failed to read initial frame from camera.")

    # 先做一次 flip，因為後面 loop 裡也會 flip
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # --- 在這裡定義 P1 / P2 的畫圖框框 ---
    mid_x = width // 2
    top_margin = 100       # 避開 HP/回合 UI
    bottom_margin = 60     # 避開底下文字

    # (x_min, y_min, x_max, y_max)
    p1_box = (0, top_margin, mid_x - 1, height - bottom_margin)
    p2_box = (mid_x, top_margin, width - 1, height - bottom_margin)

    hand_tracker = HandTracker(max_num_hands=2)

    canvas_p1 = CanvasManager(
        width=width, height=height, stroke_color=(0, 255, 0))
    canvas_p2 = CanvasManager(
        width=width, height=height, stroke_color=(255, 0, 0))

    players = {
        1: Player(player_id=1, name="Player 1", color=(0, 255, 0)),
        2: Player(player_id=2, name="Player 2", color=(255, 0, 0)),
    }
    game_state = GameState(players=players)
    game_state.start_round()

    window_name = "Air-Doodle Duel - Dual Player"

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            # 每幀都更新一次 h, w（以防相機解析度有變）
            height, width, _ = frame.shape

            now = time.perf_counter()
            game_state.update(now)

            # 1) Hand tracking for up to 2 hands
            tracked_hands = hand_tracker.process_multi(frame)

            # Assign hands to players by x position (左邊 → P1, 右邊 → P2)
            hand_infos = []  # (TrackedHand, (x, y))
            for th in tracked_hands:
                tip = hand_tracker.get_index_finger_tip(
                    th.landmarks, frame.shape)
                hand_infos.append((th, tip))

            hand_infos.sort(key=lambda item: item[1][0])  # 按 x 排序

            p1_point = None
            p2_point = None
            p1_drawing = False
            p2_drawing = False

            # P1: leftmost hand
            if len(hand_infos) >= 1:
                th, tip = hand_infos[0]
                gesture = hand_tracker.classify_gesture(th.landmarks)
                p1_point = tip
                p1_drawing = (
                    gesture == "point" and game_state.phase == Phase.DRAWING
                )
                hand_tracker.draw_hand_overlay(frame, th.landmarks)

            # P2: rightmost hand（如果有第二隻）
            if len(hand_infos) >= 2:
                th, tip = hand_infos[1]
                gesture = hand_tracker.classify_gesture(th.landmarks)
                p2_point = tip
                p2_drawing = (
                    gesture == "point" and game_state.phase == Phase.DRAWING
                )
                hand_tracker.draw_hand_overlay(frame, th.landmarks)

            # 2) 更新畫布（限制在各自的框框內）

            # --- Player 1 ---
            if p1_point is not None and p1_drawing:
                if point_in_box(p1_point, p1_box):
                    draw_pt = clamp_point_to_box(p1_point, p1_box)
                    canvas_p1.update_drawing(draw_pt, True)
                else:
                    # 手出界 → 停止畫圖（斷線）
                    canvas_p1.update_drawing((0, 0), False)
            else:
                canvas_p1.update_drawing((0, 0), False)

            # --- Player 2 ---
            if p2_point is not None and p2_drawing:
                if point_in_box(p2_point, p2_box):
                    draw_pt = clamp_point_to_box(p2_point, p2_box)
                    canvas_p2.update_drawing(draw_pt, True)
                else:
                    canvas_p2.update_drawing((0, 0), False)
            else:
                canvas_p2.update_drawing((0, 0), False)

            # 3) Overlay canvases
            frame = canvas_p1.overlay_on_frame(frame)
            frame = canvas_p2.overlay_on_frame(frame)

            # 4) FPS
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'c' to clear, 'q' or ESC to quit.",
                (10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 5) UI overlay (HP, 回合資訊, 分隔線)
            draw_ui(frame, game_state, now)

            # === 2.5) 指尖高亮（在畫布 overlay 之前畫）===
            draw_fingertip_indicator(
                frame, p1_point, p1_drawing, (0, 255, 0))   # P1 用綠色
            draw_fingertip_indicator(
                frame, p2_point, p2_drawing, (255, 0, 0))   # P2 用紅色

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c"):
                canvas_p1.reset()
                canvas_p2.reset()

    finally:
        camera.release()
        hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
