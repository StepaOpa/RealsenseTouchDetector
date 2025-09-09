"""
Модуль для обрезки изображений до зоны проекции с учетом трапецевидных искажений
Использует данные калибровки для исправления перспективы и обрезки изображения
"""

import cv2
import numpy as np
import json
import logging
from typing import Tuple, Optional, Dict, Any
import os


class ProjectionAreaCropper:
    """
    Класс для обрезки изображений до калиброванной зоны проекции
    Исправляет трапецевидные искажения и обрезает изображение
    """
    
    def __init__(self, calibration_file: str = "calibration_data.json", output_size: Tuple[int, int] = (800, 600)) -> None:
        """
        Инициализация обрезчика проекции
        
        Args:
            calibration_file: Путь к файлу калибровочных данных
            output_size: Размер выходного изображения (ширина, высота)
        """
        self.calibration_file: str = calibration_file
        self.output_width: int = output_size[0]
        self.output_height: int = output_size[1]
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Данные калибровки
        self.calibration_data: Optional[Dict[str, Any]] = None
        self.homography_matrix: Optional[np.ndarray] = None
        self.corners_2d: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        
        # Загружаем калибровочные данные
        self._load_calibration_data()
    
    def _load_calibration_data(self) -> None:
        """
        Загрузка калибровочных данных из файла
        """
        if not os.path.exists(self.calibration_file):
            self.logger.warning(f"Файл калибровки {self.calibration_file} не найден")
            return
        
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.calibration_data = data
            
            # Извлекаем необходимые данные
            calib_data = data.get('calibration_data', {})
            
            # Матрица гомографии
            homography_list = calib_data.get('homography_matrix')
            if homography_list:
                self.homography_matrix = np.array(homography_list, dtype=np.float32)
            
            # Углы проекции в 2D
            corners_2d_list = calib_data.get('corners_2d')
            if corners_2d_list:
                self.corners_2d = np.array(corners_2d_list, dtype=np.float32)
            
            if self.homography_matrix is not None and self.corners_2d is not None:
                self.is_calibrated = True
                self.logger.info("Калибровочные данные загружены успешно")
                
                # Создаем матрицу трансформации для нашего выходного размера
                self._create_transform_matrix()
            else:
                self.logger.error("Некорректные данные калибровки")
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки калибровочных данных: {e}")
    
    def _create_transform_matrix(self) -> None:
        """
        Создание матрицы трансформации для перевода в прямоугольную область
        """
        if self.corners_2d is None:
            return
        
        # Целевые точки - углы прямоугольника нужного размера
        target_points = np.array([
            [0, 0],                                          # Верхний левый
            [self.output_width - 1, 0],                      # Верхний правый
            [self.output_width - 1, self.output_height - 1], # Нижний правый
            [0, self.output_height - 1]                      # Нижний левый
        ], dtype=np.float32)
        
        # Вычисляем матрицу перспективной трансформации
        self.transform_matrix = cv2.getPerspectiveTransform(self.corners_2d, target_points)
        
        self.logger.info(f"Матрица трансформации создана для размера {self.output_width}x{self.output_height}")
    
    def crop_color_image(self, color_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Обрезка цветного изображения до зоны проекции
        
        Args:
            color_image: Входное цветное изображение
            
        Returns:
            Обрезанное и исправленное изображение или None при ошибке
        """
        if not self.is_calibrated:
            self.logger.warning("Калибровка не загружена")
            return None
        
        if self.transform_matrix is None:
            self.logger.error("Матрица трансформации не создана")
            return None
        
        try:
            # Применяем перспективную трансформацию
            cropped_image = cv2.warpPerspective(
                color_image,
                self.transform_matrix,
                (self.output_width, self.output_height)
            )
            
            return cropped_image
            
        except Exception as e:
            self.logger.error(f"Ошибка при обрезке цветного изображения: {e}")
            return None
    
    def crop_depth_image(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Обрезка изображения глубины до зоны проекции
        
        Args:
            depth_image: Входное изображение глубины
            
        Returns:
            Обрезанное и исправленное изображение глубины или None при ошибке
        """
        if not self.is_calibrated:
            self.logger.warning("Калибровка не загружена")
            return None
        
        if self.transform_matrix is None:
            self.logger.error("Матрица трансформации не создана")
            return None
        
        try:
            # Применяем перспективную трансформацию
            cropped_depth = cv2.warpPerspective(
                depth_image,
                self.transform_matrix,
                (self.output_width, self.output_height),
                flags=cv2.INTER_NEAREST  # Для глубины используем nearest neighbor
            )
            
            return cropped_depth
            
        except Exception as e:
            self.logger.error(f"Ошибка при обрезке изображения глубины: {e}")
            return None
    
    def crop_both_images(self, color_image: np.ndarray, depth_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Обрезка обоих изображений одновременно
        
        Args:
            color_image: Входное цветное изображение
            depth_image: Входное изображение глубины
            
        Returns:
            Кортеж (обрезанное_цветное, обрезанная_глубина)
        """
        cropped_color = self.crop_color_image(color_image)
        cropped_depth = self.crop_depth_image(depth_image)
        
        return cropped_color, cropped_depth
    
    def get_projection_info(self) -> Dict[str, Any]:
        """
        Получение информации о проекции
        
        Returns:
            Словарь с информацией о калибровке
        """
        if not self.is_calibrated or self.calibration_data is None:
            return {"calibrated": False}
        
        calib_data = self.calibration_data.get('calibration_data', {})
        
        return {
            "calibrated": True,
            "output_size": (self.output_width, self.output_height),
            "projection_width_meters": calib_data.get('projection_width', 0),
            "projection_height_meters": calib_data.get('projection_height', 0),
            "corners_2d": self.corners_2d.tolist() if self.corners_2d is not None else [],
            "corners_3d": calib_data.get('corners_3d', [])
        }
    
    def visualize_crop_area(self, image: np.ndarray) -> np.ndarray:
        """
        Визуализация области обрезки на исходном изображении
        
        Args:
            image: Исходное изображение
            
        Returns:
            Изображение с наложенной областью обрезки
        """
        if not self.is_calibrated or self.corners_2d is None:
            return image.copy()
        
        result = image.copy()
        
        # Рисуем контур зоны проекции
        corners_int = self.corners_2d.astype(np.int32)
        cv2.polylines(result, [corners_int], True, (0, 255, 0), 2)
        
        # Нумеруем углы
        for i, (x, y) in enumerate(corners_int):
            cv2.circle(result, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                result,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Добавляем информацию
        cv2.putText(
            result,
            f"Projection Area: {self.output_width}x{self.output_height}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return result
    
    def is_point_in_projection(self, x: int, y: int) -> bool:
        """
        Проверка, находится ли точка внутри зоны проекции
        
        Args:
            x: X координата
            y: Y координата
            
        Returns:
            True если точка внутри зоны проекции
        """
        if not self.is_calibrated or self.corners_2d is None:
            return False
        
        # Используем функцию OpenCV для проверки точки в полигоне
        point = np.array([[x, y]], dtype=np.float32)
        corners_reshaped = self.corners_2d.reshape((-1, 1, 2)).astype(np.int32)
        
        result = cv2.pointPolygonTest(corners_reshaped, (float(x), float(y)), False)
        return result >= 0  # >= 0 означает внутри или на границе
    
    def transform_point_to_cropped(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        Преобразование точки из исходного изображения в координаты обрезанного
        
        Args:
            x: X координата в исходном изображении
            y: Y координата в исходном изображении
            
        Returns:
            Координаты в обрезанном изображении или None если точка вне зоны
        """
        if not self.is_calibrated or self.transform_matrix is None:
            return None
        
        if not self.is_point_in_projection(x, y):
            return None
        
        try:
            # Преобразуем точку
            point = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.transform_matrix)
            
            tx, ty = transformed[0][0]
            
            # Проверяем, что точка попадает в границы выходного изображения
            if 0 <= tx < self.output_width and 0 <= ty < self.output_height:
                return int(tx), int(ty)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка трансформации точки: {e}")
            return None


def test_cropper() -> None:
    """
    Тест функции обрезки проекции
    """
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Создаем тестовое изображение
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 150, 200)  # Заливаем цветом
    
    # Создаем обрезчик
    cropper = ProjectionAreaCropper(output_size=(400, 300))
    
    if cropper.is_calibrated:
        logger.info("Калибровка загружена успешно")
        
        # Визуализируем область обрезки
        visualized = cropper.visualize_crop_area(test_image)
        cv2.imshow("Projection Area", visualized)
        
        # Обрезаем изображение
        cropped = cropper.crop_color_image(test_image)
        if cropped is not None:
            cv2.imshow("Cropped Image", cropped)
            logger.info(f"Обрезанное изображение: {cropped.shape}")
        
        # Показываем информацию
        info = cropper.get_projection_info()
        logger.info(f"Информация о проекции: {info}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.error("Калибровка не загружена")


if __name__ == "__main__":
    test_cropper()
