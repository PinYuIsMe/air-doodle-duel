import cv2


class Camera:
    """
    Simple wrapper around OpenCV VideoCapture.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        # CAP_DSHOW 比較適合 Windows，避免一些啟動延遲
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

        # 設定解析度（有些相機可能不完全遵守，但會盡量接近）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera device {self.device_index}")

    def read(self):
        """
        Read a single frame from the camera.

        Returns:
            success (bool), frame (np.ndarray in BGR format)
        """
        success, frame = self.cap.read()
        return success, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
