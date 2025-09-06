"""
Модуль для определения плоскости пола и сегментации объектов
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List
import math


class FloorDetector:
    """Класс для детекции пола и сегментации объектов близко к полу"""
    
    def __init__(self, floor_depth: float = 1.5, tolerance: float = 0.2) -> None:
        """
        Инициализация детектора пола
        
        Args:
            floor_depth: Расстояние до пола в метрах
            tolerance: Допустимое отклонение от пола в метрах (±)
        """
        self.floor_depth: float = floor_depth
        self.tolerance: float = tolerance
        
        # Параметры морфологических операций
        self.morphology_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.close_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Минимальные параметры контуров
        self.min_contour_area: int = 500
        self.max_contour_area: int = 50000
        
    def set_floor_depth(self, depth: float) -> None:
        """
        Установка глубины пола
        
        Args:
            depth: Глубина пола в метрах
        """
        self.floor_depth = depth
        
    def create_floor_mask(self, depth_image: np.ndarray, depth_scale: float) -> np.ndarray:
        """
        Создание маски пола на основе глубины
        
        Args:
            depth_image: Изображение глубины
            depth_scale: Масштаб глубины (для преобразования в метры)
            
        Returns:
            Бинарная маска пола
        """
        # Проверка корректности изображения глубины
        if depth_image is None or depth_image.size == 0:
            return np.zeros((480, 640), dtype=np.uint8)
        
        # Преобразование depth в метры
        depth_meters = depth_image.astype(np.float32) * depth_scale
        
        # Создание маски для области пола (с учетом допуска)
        floor_min = float(self.floor_depth - self.tolerance)
        floor_max = float(self.floor_depth + self.tolerance)
        
        # Проверка корректности диапазона
        if floor_min <= 0 or floor_max <= 0 or floor_min >= floor_max:
            return np.zeros(depth_image.shape, dtype=np.uint8)
        
        # Маска пола - области в допустимом диапазоне глубины
        floor_mask = cv2.inRange(depth_meters, floor_min, floor_max)
        
        # Убираем нулевые значения (недоступные пиксели)
        valid_mask = depth_image > 0
        floor_mask = cv2.bitwise_and(floor_mask, floor_mask, mask=valid_mask.astype(np.uint8))
        
        return floor_mask
    
    def detect_objects_on_floor(self, depth_image: np.ndarray, depth_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция объектов на полу (ноги)
        
        Args:
            depth_image: Изображение глубины
            depth_scale: Масштаб глубины
            
        Returns:
            Tuple с маской объектов и отфильтрованной маской
        """
        # Проверка корректности изображения глубины
        if depth_image is None or depth_image.size == 0:
            empty_mask = np.zeros((480, 640), dtype=np.uint8)
            return empty_mask, empty_mask
        
        # Преобразование depth в метры
        depth_meters = depth_image.astype(np.float32) * depth_scale
        
        # Определение диапазона для поиска объектов на полу
        # Ищем объекты которые выше пола, но не слишком высоко
        floor_level = self.floor_depth
        object_min_height = float(floor_level - self.tolerance)  # Немного ниже пола
        object_max_height = float(floor_level - 0.05)  # На 5 см выше пола
        
        # Проверка корректности диапазона
        if object_min_height <= 0 or object_max_height <= 0 or object_min_height >= object_max_height:
            empty_mask = np.zeros(depth_image.shape, dtype=np.uint8)
            return empty_mask, empty_mask
        
        # Создание маски для объектов
        objects_mask = cv2.inRange(depth_meters, object_min_height, object_max_height)
        
        # Убираем нулевые значения
        valid_mask = depth_image > 0
        objects_mask = cv2.bitwise_and(objects_mask, objects_mask, mask=valid_mask.astype(np.uint8))
        
        # Морфологическая обработка для удаления шумов и заполнения пробелов
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, self.morphology_kernel)
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, self.close_kernel)
        
        # Дополнительная фильтрация - медианный фильтр
        objects_mask_filtered = cv2.medianBlur(objects_mask, 5)
        
        return objects_mask, objects_mask_filtered
    
    def find_foot_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Поиск контуров ног на маске
        
        Args:
            mask: Бинарная маска объектов
            
        Returns:
            Список контуров ног
        """
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтрация контуров по площади
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                # Дополнительная проверка - контур должен быть достаточно компактным
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = (perimeter * perimeter) / area
                    if compactness < 50:  # Фильтр по компактности
                        valid_contours.append(contour)
                        
        return valid_contours
    
    def get_contour_centers(self, contours: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Вычисление центров контуров
        
        Args:
            contours: Список контуров
            
        Returns:
            Список координат центров (x, y)
        """
        centers = []
        for contour in contours:
            # Вычисление моментов для нахождения центра
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centers.append((cx, cy))
                
        return centers
    
    def create_debug_image(self, 
                          color_image: np.ndarray, 
                          mask: np.ndarray, 
                          contours: List[np.ndarray],
                          centers: List[Tuple[int, int]]) -> np.ndarray:
        """
        Создание изображения для отладки
        
        Args:
            color_image: Исходное цветное изображение
            mask: Маска объектов
            contours: Контуры ног
            centers: Центры контуров
            
        Returns:
            Изображение с наложенной отладочной информацией
        """
        # Копируем исходное изображение
        debug_image = color_image.copy()
        
        # Наложение маски (полупрозрачно)
        if mask.shape[:2] == debug_image.shape[:2]:
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            debug_image = cv2.addWeighted(debug_image, 0.7, mask_colored, 0.3, 0)
        else:
            # Изменение размера маски под изображение
            mask_resized = cv2.resize(mask, (debug_image.shape[1], debug_image.shape[0]))
            mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_HOT)
            debug_image = cv2.addWeighted(debug_image, 0.7, mask_colored, 0.3, 0)
        
        # Рисование контуров
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
        
        # Рисование центров
        for i, (cx, cy) in enumerate(centers):
            # Центр стопы - красный кружок
            cv2.circle(debug_image, (cx, cy), 8, (0, 0, 255), -1)
            # Номер стопы
            cv2.putText(debug_image, f"Foot {i+1}", (cx-20, cy-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(debug_image, f"({cx}, {cy})", (cx-30, cy+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug_image
    
    def auto_detect_floor(self, depth_image: np.ndarray, depth_scale: float, 
                         sample_region: Tuple[int, int, int, int] = None) -> float:
        """
        Автоматическое определение глубины пола
        
        Args:
            depth_image: Изображение глубины
            depth_scale: Масштаб глубины
            sample_region: Регион для выборки (x, y, width, height), если None - центр изображения
            
        Returns:
            Определенная глубина пола в метрах
        """
        height, width = depth_image.shape
        
        # Если регион не задан, используем центральную область
        if sample_region is None:
            x = width // 4
            y = height // 2
            w = width // 2
            h = height // 4
        else:
            x, y, w, h = sample_region
            
        # Извлекаем область для анализа
        roi = depth_image[y:y+h, x:x+w]
        
        # Преобразуем в метры и убираем нулевые значения
        roi_meters = roi.astype(np.float32) * depth_scale
        valid_depths = roi_meters[roi_meters > 0]
        
        if len(valid_depths) > 0:
            # Используем медиану для устойчивости к выбросам
            floor_depth = np.median(valid_depths)
            self.set_floor_depth(floor_depth)
            return floor_depth
            
        return self.floor_depth


def test_floor_detector() -> None:
    """Тестирование детектора пола"""
    from realsense_camera import RealSenseManager
    
    camera = RealSenseManager()
    detector = FloorDetector()
    
    if not camera.start():
        return
        
    try:
        print("Нажмите 'c' для автоматической калибровки пола")
        print("Нажмите ESC для выхода")
        
        while True:
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                continue
                
            # Детекция объектов на полу
            objects_mask, filtered_mask = detector.detect_objects_on_floor(
                depth_image, camera.depth_scale
            )
            
            # Поиск контуров ног
            contours = detector.find_foot_contours(filtered_mask)
            centers = detector.get_contour_centers(contours)
            
            # Создание отладочного изображения
            debug_image = detector.create_debug_image(
                color_image, filtered_mask, contours, centers
            )
            
            # Отображение результатов
            cv2.imshow('Floor Detection - Debug', debug_image)
            cv2.imshow('Floor Detection - Mask', filtered_mask)
            
            # Информация на экране
            info_text = f"Floor: {detector.floor_depth:.2f}m, Feet: {len(centers)}"
            cv2.putText(debug_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):  # Калибровка пола
                floor_depth = detector.auto_detect_floor(depth_image, camera.depth_scale)
                print(f"Автоматически определена глубина пола: {floor_depth:.3f} м")
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_floor_detector()
