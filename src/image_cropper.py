import cv2
import numpy as np


class ImageCropper:
    def __init__(
        self, image: np.ndarray = None, output_size: tuple[int, int] = (640, 480)
    ):
        self.image = image
        self.output_size = output_size
        self.points = []

        self.transformed_image = None

    def select_crop_region(self):
        """Выбор региона для кроппинга."""
        self.points = []
        self.window_name = "Select Region"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)

    def crop_image(self, image: np.ndarray) -> np.ndarray:

        if len(self.points) != 4:
            print("Error: Exactly 4 points are required for cropping.")
            return None

        src_points = np.float32(self.points)
        dst_points = np.float32(
            [
                [0, 0],
                [self.output_size[0], 0],
                [self.output_size[0], self.output_size[1]],
                [0, self.output_size[1]],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.transformed_image = cv2.warpPerspective(image, matrix, self.output_size)

        return self.transformed_image

    def _mouse_callback(self, event, x, y, flags, param):
        """Обработчик кликов мыши."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.image)
            print(f"Point {len(self.points)} selected: ({x}, {y})")
