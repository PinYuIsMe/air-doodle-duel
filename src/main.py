import cv2
import numpy as np


def main():
    # 建一個黑色畫布當作 placeholder
    width, height = 960, 540
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    window_name = "Air-Doodle Duel"

    while True:
        cv2.imshow(window_name, canvas)

        # 每 1ms 檢查鍵盤輸入，按 q 或 ESC 離開
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
