"""
Калибровка зоны проекции для системы распознавания касаний
Позволяет пользователю обозначить углы проекции и сохранить калибровочные данные
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from realsense_camera import RealSenseCamera
import pyrealsense2 as rs


class ProjectionCalibrator:
    """
    Класс для калибровки зоны проекции
    """
    
    def __init__(self, camera: RealSenseCamera) -> None:
        """
        Инициализация калибратора
        
        Args:
            camera: Экземпляр камеры RealSense
        """
        self.camera: RealSenseCamera = camera
        self.corners_2d: List[Tuple[int, int]] = []
        self.corners_3d: List[Tuple[float, float, float]] = []
        self.current_frame: Optional[np.ndarray] = None
        self.current_depth: Optional[np.ndarray] = None
        self.depth_scale: float = 0.0
        self.color_intrinsics: Optional[rs.intrinsics] = None
        
        # Настройка логирования
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Параметры отображения
        self.window_name: str = "Калибровка проекции - нажмите на углы"
        self.target_corners: int = 4
        
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Обработчик событий мыши для выбора углов проекции
        
        Args:
            event: Тип события мыши
            x: X координата курсора
            y: Y координата курсора
            flags: Дополнительные флаги
            param: Дополнительные параметры
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners_2d) < self.target_corners:
            # Добавляем 2D координату
            self.corners_2d.append((x, y))
            
            # Получаем 3D координату из depth изображения
            if self.current_depth is not None and self.color_intrinsics is not None:
                world_coords = self._pixel_to_world(x, y)
                if world_coords is not None:
                    self.corners_3d.append(world_coords)
                    self.logger.info(f"Угол {len(self.corners_2d)}: 2D({x}, {y}) -> 3D({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})")
                else:
                    # Если не удалось получить 3D координату, удаляем 2D
                    self.corners_2d.pop()
                    self.logger.warning(f"Не удалось получить глубину для точки ({x}, {y})")
    
    def _pixel_to_world(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """
        Преобразование пиксельных координат в мировые координаты
        
        Args:
            x: X координата пикселя
            y: Y координата пикселя
            
        Returns:
            Мировые координаты (x, y, z) или None если преобразование невозможно
        """
        if self.current_depth is None or self.color_intrinsics is None:
            return None
        
        # Получаем значение глубины в пикселе
        depth_value = self.current_depth[y, x]
        if depth_value == 0:
            return None
        
        # Преобразуем в метры
        depth_meters = depth_value * self.depth_scale
        
        # Преобразуем в мировые координаты
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, [x, y], depth_meters
        )
        
        return (point_3d[0], point_3d[1], point_3d[2])
    
    def _draw_overlay(self, image: np.ndarray) -> np.ndarray:
        """
        Отрисовка интерфейса калибровки поверх изображения
        
        Args:
            image: Исходное изображение
            
        Returns:
            Изображение с наложенным интерфейсом
        """
        overlay = image.copy()
        
        # Отрисовываем уже выбранные углы
        for i, (x, y) in enumerate(self.corners_2d):
            cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                f'{i+1}',
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Если выбрано больше 1 угла, соединяем их линиями
        if len(self.corners_2d) > 1:
            for i in range(len(self.corners_2d)):
                start_point = self.corners_2d[i]
                end_point = self.corners_2d[(i + 1) % len(self.corners_2d)]
                cv2.line(overlay, start_point, end_point, (255, 0, 0), 2)
        
        # Инструкции
        instructions = [
            f"Select projection corners: {len(self.corners_2d)}/{self.target_corners}",
            "LBM - choose corner",
            "R - reset",
            "S - save (after selecting all corners)",
            "Q - exit"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                overlay,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 25
        
        return overlay
    
    def _calculate_rectangle_transform(self) -> Dict[str, Any]:
        """
        Вычисление трансформации четырехугольника в прямоугольник
        
        Returns:
            Словарь с данными трансформации
        """
        if len(self.corners_3d) != 4:
            raise ValueError("Требуется ровно 4 угла для калибровки")
        
        # Сортируем углы по часовой стрелке, начиная с верхнего левого
        corners_3d_array = np.array(self.corners_3d, dtype=np.float32)
        corners_2d_array = np.array(self.corners_2d, dtype=np.float32)
        
        # Находим центр четырехугольника
        center_3d = np.mean(corners_3d_array, axis=0)
        center_2d = np.mean(corners_2d_array, axis=0)
        
        # Сортируем углы по углу относительно центра
        def angle_from_center(point: np.ndarray, center: np.ndarray) -> float:
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        angles_2d = [angle_from_center(corner, center_2d) for corner in corners_2d_array]
        sorted_indices = sorted(range(4), key=lambda i: angles_2d[i])
        
        sorted_corners_3d = corners_3d_array[sorted_indices]
        sorted_corners_2d = corners_2d_array[sorted_indices]
        
        # Вычисляем размеры проекции в реальном мире
        width_top = np.linalg.norm(sorted_corners_3d[1] - sorted_corners_3d[0])
        width_bottom = np.linalg.norm(sorted_corners_3d[2] - sorted_corners_3d[3])
        height_left = np.linalg.norm(sorted_corners_3d[3] - sorted_corners_3d[0])
        height_right = np.linalg.norm(sorted_corners_3d[2] - sorted_corners_3d[1])
        
        # Берем средние значения для прямоугольника
        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2
        
        # Создаем целевой прямоугольник (нормализованные координаты 0-1)
        target_rectangle = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)
        
        # Вычисляем матрицу гомографии для 2D трансформации
        homography_matrix = cv2.getPerspectiveTransform(sorted_corners_2d, target_rectangle * 1000)
        
        return {
            "corners_3d": sorted_corners_3d.tolist(),
            "corners_2d": sorted_corners_2d.tolist(),
            "center_3d": center_3d.tolist(),
            "projection_width": float(avg_width),
            "projection_height": float(avg_height),
            "homography_matrix": homography_matrix.tolist(),
            "target_rectangle": target_rectangle.tolist()
        }
    
    def calibrate(self) -> bool:
        """
        Запуск процесса калибровки
        
        Returns:
            True если калибровка прошла успешно, False в противном случае
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.logger.info("Начинаем калибровку проекции")
        self.logger.info("Выберите 4 угла проекции, начиная с верхнего левого по часовой стрелке")
        
        # Получаем параметры камеры
        self.depth_scale = self.camera.get_depth_scale()
        color_intrinsics, _ = self.camera.get_intrinsics()
        self.color_intrinsics = color_intrinsics
        
        while True:
            # Получаем кадры с камеры
            color_image, depth_image = self.camera.get_frames()
            
            if color_image is None or depth_image is None:
                self.logger.warning("Не удалось получить кадры с камеры")
                continue
            
            self.current_frame = color_image
            self.current_depth = depth_image
            
            # Создаем изображение с интерфейсом
            display_image = self._draw_overlay(color_image)
            
            # Отображаем изображение
            cv2.imshow(self.window_name, display_image)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.logger.info("Калибровка отменена пользователем")
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):
                # Сброс выбранных углов
                self.corners_2d.clear()
                self.corners_3d.clear()
                self.logger.info("Углы сброшены")
            elif key == ord('s') and len(self.corners_2d) == self.target_corners:
                # Сохранение калибровки
                try:
                    calibration_data = self._calculate_rectangle_transform()
                    self._save_calibration_data(calibration_data)
                    self.logger.info("Калибровка завершена успешно")
                    cv2.destroyAllWindows()
                    return True
                except Exception as e:
                    self.logger.error(f"Ошибка при сохранении калибровки: {e}")
        
        cv2.destroyAllWindows()
        return False
    
    def _save_calibration_data(self, calibration_data: Dict[str, Any]) -> None:
        """
        Сохранение данных калибровки в файл
        
        Args:
            calibration_data: Данные калибровки
        """
        filename = "calibration_data.json"
        
        # Добавляем метаданные
        final_data = {
            "version": "1.0",
            "calibration_type": "projection_area",
            "camera_info": {
                "depth_scale": self.depth_scale,
                "color_intrinsics": {
                    "width": self.color_intrinsics.width if self.color_intrinsics else 0,
                    "height": self.color_intrinsics.height if self.color_intrinsics else 0,
                    "fx": self.color_intrinsics.fx if self.color_intrinsics else 0,
                    "fy": self.color_intrinsics.fy if self.color_intrinsics else 0,
                    "ppx": self.color_intrinsics.ppx if self.color_intrinsics else 0,
                    "ppy": self.color_intrinsics.ppy if self.color_intrinsics else 0,
                }
            },
            "calibration_data": calibration_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Данные калибровки сохранены в {filename}")


def main() -> None:
    """
    Основная функция для запуска калибровки проекции
    """
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Создаем камеру
    camera = RealSenseCamera(width=640, height=480, fps=30)
    
    try:
        with camera:
            logger.info("Камера запущена для калибровки")
            
            # Создаем калибратор
            calibrator = ProjectionCalibrator(camera)
            
            # Запускаем калибровку
            success = calibrator.calibrate()
            
            if success:
                logger.info("Калибровка завершена успешно!")
            else:
                logger.info("Калибровка была отменена")
                
    except Exception as e:
        logger.error(f"Ошибка при работе с камерой: {e}")


if __name__ == "__main__":
    main()
