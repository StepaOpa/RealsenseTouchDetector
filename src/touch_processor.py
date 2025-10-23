import cv2
import numpy as np
import pyrealsense2 as rs
from typing import List, Tuple, Optional
import logging


class TouchProcessor:
    def __init__(self):
        # Плоскость: Ax + By + Cz + D = 0
        self.plane: Optional[Tuple[float, float, float, float]] = None  # (A, B, C, D)
        self.plane_normal: Optional[np.ndarray] = (
            None  # нормализованный вектор (a, b, c)
        )
        self.plane_d: Optional[float] = None  # D

        # Параметры фильтрации
        self.min_height_m: float = 0.0
        self.max_height_m: float = 0.15
        self.min_area_px: int = 100
        self.max_area_px: int = 5000

        # Для визуализации
        self.binary_image: Optional[np.ndarray] = None
        self.contours: List[np.ndarray] = []
        self.window_name: str = "Binary Touch Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Интринсики
        self.intrinsics: Optional[rs.intrinsics] = None

        # Логирование
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_floor_plane(self, plane_model: Tuple[float, float, float, float]):
        """
        Устанавливает уравнение плоскости: Ax + By + Cz + D = 0.
        Нормаль (A, B, C) должна быть нормализована.
        """
        A, B, C, D = plane_model
        norm = np.linalg.norm([A, B, C])
        if norm == 0:
            raise ValueError("Нормаль плоскости не может быть нулевой.")
        self.plane_normal = np.array([A, B, C]) / norm
        self.plane_d = D
        self.plane = (A, B, C, D)
        self.logger.info(
            f"Плоскость установлена: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0"
        )

    def set_intrinsics(self, intrinsics: rs.intrinsics):
        """Устанавливает внутренние параметры камеры."""
        self.intrinsics = intrinsics

    def set_height_range(self, min_height_m: float, max_height_m: float):
        """Устанавливает диапазон высоты над плоскостью (в метрах)."""
        self.min_height_m = min_height_m
        self.max_height_m = max_height_m

    def set_area_range(self, min_area_px: int, max_area_px: int):
        """Устанавливает диапазон площади контуров (в пикселях)."""
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

    def calculate_distance_from_point_to_plane(
        self, point_3d: Tuple[float, float, float]
    ) -> float:
        """
        Вычисляет расстояние от 3D-точки до плоскости.
        Возвращает signed distance.
        """
        if self.plane_normal is None or self.plane_d is None:
            raise RuntimeError("Плоскость не установлена.")
        x, y, z = point_3d
        p = np.array([x, y, z])
        return np.dot(self.plane_normal, p) + self.plane_d

    def process_frame(self, depth_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Обрабатывает кадр глубины и возвращает координаты центров объектов над плоскостью.

        Args:
            depth_frame: Кадр глубины (uint16, мм).

        Returns:
            Список кортежей (x, y) — координаты центров объектов в пикселях кадра.
        """
        if self.plane is None:
            raise RuntimeError("Плоскость не установлена. Вызовите set_floor_plane().")
        if self.intrinsics is None:
            raise RuntimeError("Интринсики не установлены. Вызовите set_intrinsics().")

        h, w = depth_frame.shape
        binary_map = np.zeros((h, w), dtype=np.uint8)

        # Фильтруем глубину
        valid = depth_frame > 0
        if not np.any(valid):
            self.binary_image = binary_map
            self.contours = []
            cv2.imshow(self.window_name, self.binary_image)
            cv2.waitKey(1)
            return []

        # Создаём сетку координат
        v_coords, u_coords = np.mgrid[0:h, 0:w]
        u_flat = u_coords[valid].ravel()
        v_flat = v_coords[valid].ravel()
        depth_flat = depth_frame[valid].ravel()

        # Векторизованная депроецировка (создаём список пикселей -> 3D точки)
        # Это самая медленная часть, но векторизовать её полностью сложно через rs2
        # Вместо этого, можно ускорить, если использовать только центральные пиксели или subsample
        points_3d = np.array(
            [
                rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], d * 0.001)
                for u, v, d in zip(u_flat, v_flat, depth_flat)
            ]
        )

        if points_3d.size == 0:
            self.binary_image = binary_map
            self.contours = []
            cv2.imshow(self.window_name, self.binary_image)
            cv2.waitKey(1)
            return []

        # Векторизованное расстояние до плоскости
        distances = np.dot(points_3d, self.plane_normal) + self.plane_d

        # Фильтруем по высоте
        mask_height = (distances >= self.min_height_m) & (
            distances <= self.max_height_m
        )
        u_filtered = u_flat[mask_height]
        v_filtered = v_flat[mask_height]

        binary_map[v_filtered, u_filtered] = 255

        # Морфология (минималистичная)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)

        # Контуры
        contours, _ = cv2.findContours(
            binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Фильтрация по площади и центры
        touch_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area_px <= area <= self.max_area_px:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    touch_points.append((cx, cy))

        # Сохраняем для визуализации
        self.binary_image = binary_map
        self.contours = contours

        # Визуализация
        cv2.imshow(self.window_name, self.binary_image)
        cv2.waitKey(1)

        return touch_points

    def get_binary_image(self) -> Optional[np.ndarray]:
        """Возвращает последнее бинарное изображение."""
        return self.binary_image

    def get_contours(self) -> List[np.ndarray]:
        """Возвращает последние найденные контуры."""
        return self.contours

    def close(self):
        """Закрывает окно визуализации."""
        cv2.destroyWindow(self.window_name)
