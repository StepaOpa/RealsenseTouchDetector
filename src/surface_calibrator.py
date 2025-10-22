import pyrealsense2 as rs
import numpy as np
from typing import List, Tuple, Optional
from realsense_camera import RealSenseCamera
import open3d as o3d


class SurfaceCalibrator:
    def __init__(self, camera: "RealSenseCamera"):
        """
        Инициализация калибровщика плоскости.

        Args:
            camera: Экземпляр вашего RealSenseCamera (уже запущенный).
        """
        self.camera = camera
        self.points_2d: List[Tuple[int, int]] = []  # [(u, v), ...]
        self.points_3d: List[Tuple[float, float, float]] = (
            []
        )  # [(X, Y, Z), ...] в метрах

    def set_2d_points(self, points_2d: List[Tuple[int, int]]):
        """
        Устанавливает 4 точки в 2D-координатах (пиксели).

        Args:
            points_2d: Список из 4 кортежей (u, v). Порядок не важен для 3D-конвертации.
        """
        if len(points_2d) != 4:
            raise ValueError("Требуется ровно 4 точки.")
        self.points_2d = points_2d
        self.points_3d = []  # сбросим 3D-точки

    def compute_3d_points(self) -> List[Tuple[float, float, float]]:
        """
        Преобразует сохранённые 2D-точки в 3D-координаты с использованием текущих данных глубины.

        Returns:
            Список из 4 точек в формате (X, Y, Z) в метрах.
            Если глубина недоступна в какой-то точке — возвращает (0, 0, 0) для неё.
        """
        if not self.points_2d:
            raise RuntimeError("2D-точки не заданы. Вызовите set_2d_points().")

        # Получаем текущие кадры глубины и цвета (нужен depth_frame для данных глубины)
        color_image, depth_image = self.camera.get_frames()
        if depth_image is None:
            raise RuntimeError("Не удалось получить изображение глубины.")

        # Получаем внутренние параметры цветной камеры
        color_intrinsics, _ = self.camera.get_intrinsics()
        if color_intrinsics is None:
            raise RuntimeError("Не удалось получить внутренние параметры камеры.")

        depth_scale = self.camera.get_depth_scale()
        if depth_scale == 0.0:
            raise RuntimeError("Некорректный масштаб глубины.")

        self.points_3d = []
        h, w = depth_image.shape

        for u, v in self.points_2d:
            # Проверка границ
            if not (0 <= u < w and 0 <= v < h):
                print(f"Предупреждение: точка ({u}, {v}) вне границ изображения.")
                self.points_3d.append((0.0, 0.0, 0.0))
                continue

            # Получаем значение глубины (в "сырых" единицах)
            depth_raw = depth_image[v, u]
            if depth_raw == 0:
                print(f"Предупреждение: нет данных глубины в точке ({u}, {v}).")
                self.points_3d.append((0.0, 0.0, 0.0))
                continue

            # Переводим в метры
            depth_m = depth_raw * depth_scale

            # Депроекция в 3D
            x, y, z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [u, v], depth_m)
            self.points_3d.append((x, y, z))

        return self.points_3d

    def get_3d_points(self) -> List[Tuple[float, float, float]]:
        """Возвращает уже вычисленные 3D-точки (если есть)."""
        return self.points_3d.copy()

    def estimate_floor_plane_with_ransac(
        self,
        roi_radius: int = 50,
        distance_threshold: float = 0.02,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Оценивает плоскость пола с помощью RANSAC (Open3D), используя регион вокруг 4 заданных точек.

        Args:
            roi_radius: Радиус (в пикселях) вокруг каждой 2D-точки для сбора точек облака.
            distance_threshold: Порог расстояния до плоскости (в метрах) для inliers.
            ransac_n: Количество точек для генерации гипотезы (3 для плоскости).
            num_iterations: Число итераций RANSAC.

        Returns:
            Уравнение плоскости (A, B, C, D), где Ax + By + Cz + D = 0.
            None, если не удалось оценить.
        """
        if not self.points_2d:
            raise RuntimeError("2D-точки не заданы. Вызовите set_2d_points().")

        # Получаем кадры
        color_image, depth_image = self.camera.get_frames()
        if depth_image is None:
            raise RuntimeError("Не удалось получить изображение глубины.")

        color_intrinsics, _ = self.camera.get_intrinsics()
        depth_scale = self.camera.get_depth_scale()

        h, w = depth_image.shape
        all_points_3d = []

        # Собираем точки в окрестности каждой из 4 точек
        for u_center, v_center in self.points_2d:
            u_min = max(0, u_center - roi_radius)
            u_max = min(w, u_center + roi_radius)
            v_min = max(0, v_center - roi_radius)
            v_max = min(h, v_center + roi_radius)

            # Извлекаем подобласть глубины
            depth_roi = depth_image[v_min:v_max, u_min:u_max]
            if depth_roi.size == 0:
                continue

            # Генерируем сетку координат пикселей
            v_coords, u_coords = np.mgrid[v_min:v_max, u_min:u_max]
            u_flat = u_coords.ravel()
            v_flat = v_coords.ravel()
            depth_flat = depth_roi.ravel()

            # Фильтруем недействительные глубины
            valid = (depth_flat > 0) & (
                depth_flat < 65535
            )  # исключаем 0 и макс. значения
            u_valid = u_flat[valid]
            v_valid = v_flat[valid]
            depth_valid = depth_flat[valid] * depth_scale

            # Депроекция
            points_3d = np.array(
                [
                    rs.rs2_deproject_pixel_to_point(color_intrinsics, [u, v], d)
                    for u, v, d in zip(u_valid, v_valid, depth_valid)
                ]
            )

            all_points_3d.append(points_3d)

        if not all_points_3d:
            print("Ошибка: не удалось собрать 3D-точки для RANSAC.")
            return None

        # Объединяем все точки
        cloud_points = np.vstack(all_points_3d)
        if len(cloud_points) < ransac_n:
            print("Ошибка: недостаточно точек для RANSAC.")
            return None

        # Создаём облако точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)

        # Применяем RANSAC
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )
            self.plane_model = tuple(plane_model)  # (A, B, C, D)
            print(f"Плоскость пола найдена: {self.plane_model}")
            return self.plane_model
        except Exception as e:
            print(f"Ошибка при выполнении RANSAC: {e}")
            return None

    def get_plane_model(self) -> Optional[Tuple[float, float, float, float]]:
        """Возвращает последнюю оценённую модель плоскости (A, B, C, D)."""
        return self.plane_model

    def distance_to_plane(self, point_3d: Tuple[float, float, float]) -> float:
        """
        Вычисляет расстояние от 3D-точки до оценённой плоскости.

        Args:
            point_3d: (x, y, z) в метрах.

        Returns:
            Расстояние в метрах (со знаком).
        """
        if self.plane_model is None:
            raise RuntimeError(
                "Плоскость не оценена. Вызовите estimate_floor_plane_with_ransac()."
            )
        A, B, C, D = self.plane_model
        x, y, z = point_3d
        return A * x + B * y + C * z + D  # нормаль уже нормирована в Open3D
