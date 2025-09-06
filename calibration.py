"""
Модуль для ручной калибровки границ проекции и преобразования координат
"""
import cv2
import numpy as np
import json
from typing import List, Tuple, Optional
import os
import math
from scipy.spatial import ConvexHull


class CalibrationManager:
    """Класс для управления калибровкой проекции"""
    
    def __init__(self, target_width: int = 1920, target_height: int = 1080, 
                 camera_width: int = 640, camera_height: int = 480) -> None:
        """
        Инициализация менеджера калибровки
        
        Args:
            target_width: Целевая ширина экранного пространства
            target_height: Целевая высота экранного пространства
            camera_width: Ширина изображения камеры
            camera_height: Высота изображения камеры
        """
        self.target_width: int = target_width
        self.target_height: int = target_height
        self.camera_width: int = camera_width
        self.camera_height: int = camera_height
        
        # Точки калибровки (углы проекционной области на изображении с камеры)
        self.calibration_points: List[Tuple[int, int]] = []
        
        # Матрица гомографии
        self.homography_matrix: Optional[np.ndarray] = None
        
        # Обратная матрица гомографии
        self.inverse_homography_matrix: Optional[np.ndarray] = None
        
        # Флаг завершения калибровки
        self.is_calibrated: bool = False
        
        # Файл для сохранения калибровки
        self.calibration_file: str = "calibration.json"
        
    def reset_calibration(self) -> None:
        """Сброс калибровки"""
        self.calibration_points = []
        self.homography_matrix = None
        self.inverse_homography_matrix = None
        self.is_calibrated = False
        
    def add_calibration_point(self, x: int, y: int) -> bool:
        """
        Добавление точки калибровки
        
        Args:
            x: Координата X на изображении камеры
            y: Координата Y на изображении камеры
            
        Returns:
            True если точка добавлена успешно, False если уже достаточно точек
        """
        if len(self.calibration_points) < 4:
            self.calibration_points.append((x, y))
            print(f"Добавлена точка калибровки #{len(self.calibration_points)}: ({x}, {y})")
            
            if len(self.calibration_points) == 4:
                self._calculate_homography()
                return True
            return True
        return False
    
    def _calculate_homography(self) -> None:
        """Улучшенное вычисление матрицы гомографии"""
        if len(self.calibration_points) != 4:
            print("Ошибка: Необходимо 4 точки для калибровки")
            return

        # Проверка выпуклости
        points = np.array(self.calibration_points)
        hull = ConvexHull(points)
        if len(hull.vertices) != 4:
            print("Точки не образуют выпуклый четырехугольник!")
            return

        # Упорядочивание точек
        center = np.mean(points, axis=0)
        points_sorted = sorted(points, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0]))
        
        src_points = np.float32(points_sorted)
        dst_points = np.float32([
            [0, 0],
            [self.target_width-1, 0],
            [self.target_width-1, self.target_height-1],
            [0, self.target_height-1]
        ])

        try:
            self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.inverse_homography_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            self.is_calibrated = True
            print("✅ Калибровка завершена успешно!")
            print(f"Целевое разрешение: {self.target_width}x{self.target_height}")
        except Exception as e:
            print(f"Ошибка вычисления гомографии: {e}")
            self.is_calibrated = False
    
    def transform_point(self, x: int, y: int, debug: bool = False) -> Tuple[int, int]:
        """
        Преобразование точки из координат камеры в экранные координаты
        
        Args:
            x: Координата X на изображении камеры
            y: Координата Y на изображении камеры
            debug: Включить отладочный вывод
            
        Returns:
            Преобразованные координаты (screen_x, screen_y)
        """
        # Проверяем корректность размеров камеры
        if self.camera_width <= 0 or self.camera_height <= 0:
            if debug:
                print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Размеры камеры не установлены! Использую значения по умолчанию.")
            self.camera_width = 640
            self.camera_height = 480
        
        if not self.is_calibrated or self.homography_matrix is None:
            # Если нет калибровки, возвращаем масштабированные координаты
            if debug:
                print(f"📏 Использование простого масштабирования: камера({x}, {y}) -> экран")
            
            result = self._simple_scale(x, y)
            
            if debug:
                print(f"   Результат: {result}")
            return result
            
        # Применение гомографии
        point = np.float32([[[x, y]]])
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        raw_x = transformed[0][0][0] 
        raw_y = transformed[0][0][1]
        
        screen_x = int(np.clip(raw_x, 0, self.target_width - 1))
        screen_y = int(np.clip(raw_y, 0, self.target_height - 1))
        
        if debug:
            print(f"🔄 Гомография: камера({x}, {y}) -> экран({screen_x}, {screen_y})")
        
        return self._simple_scale(x, y)
    
    def is_point_in_projection_area(self, x: int, y: int) -> bool:
        """
        Проверяет, находится ли точка внутри области проекции (калибровки)
        
        Args:
            x: Координата X на изображении камеры
            y: Координата Y на изображении камеры
            
        Returns:
            True если точка внутри области проекции, False иначе
        """
        if not self.is_calibrated or len(self.calibration_points) != 4:
            # Если калибровка не выполнена, считаем что вся область доступна
            return True
        
        # Создаем контур области калибровки
        contour = np.array(self.calibration_points, dtype=np.int32)
        
        # Используем OpenCV для проверки принадлежности точки к полигону
        result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
        
        # result >= 0 означает, что точка внутри или на границе полигона
        return result >= 0
    
    def filter_points_in_projection_area(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Фильтрует список точек, оставляя только те, что находятся в области проекции
        
        Args:
            points: Список точек [(x, y), ...]
            
        Returns:
            Отфильтрованный список точек
        """
        if not self.is_calibrated or len(self.calibration_points) != 4:
            return points
        
        filtered_points = []
        for x, y in points:
            if self.is_point_in_projection_area(x, y):
                filtered_points.append((x, y))
        
        return filtered_points
    
    def _simple_scale(self, x: int, y: int) -> Tuple[int, int]:
        """
        Простое масштабирование без гомографии (fallback)
        
        Args:
            x: Координата X на изображении камеры
            y: Координата Y на изображении камеры
            
        Returns:
            Масштабированные координаты
        """
        # Используем реальные размеры камеры, с fallback значениями
        camera_width = self.camera_width if self.camera_width > 0 else 640
        camera_height = self.camera_height if self.camera_height > 0 else 480
        
        # Проверяем границы входных координат
        x_clamped = max(0, min(x, camera_width - 1))
        y_clamped = max(0, min(y, camera_height - 1))
        
        # Простое линейное масштабирование
        scale_x = self.target_width / camera_width
        scale_y = self.target_height / camera_height
        
        screen_x = int(x_clamped * scale_x)
        screen_y = int(y_clamped * scale_y)
        
        # Проверяем границы выходных координат
        screen_x_final = max(0, min(screen_x, self.target_width - 1))
        screen_y_final = max(0, min(screen_y, self.target_height - 1))
        
        return screen_x_final, screen_y_final
    
    def set_camera_resolution(self, width: int, height: int) -> None:
        """
        Установка реального разрешения камеры
        
        Args:
            width: Ширина изображения камеры
            height: Высота изображения камеры
        """
        self.camera_width = width
        self.camera_height = height
        print(f"Установлено разрешение камеры: {width}x{height}")
        
        # Вычисляем коэффициенты масштабирования для справки
        if width > 0 and height > 0:
            scale_x = self.target_width / width
            scale_y = self.target_height / height
            print(f"Коэффициенты масштабирования: X={scale_x:.2f}, Y={scale_y:.2f}")
    
    def save_calibration(self, filename: str = None) -> bool:
        """
        Сохранение калибровки в файл
        
        Args:
            filename: Имя файла для сохранения
            
        Returns:
            True если сохранение прошло успешно
        """
        if filename is None:
            filename = self.calibration_file
            
        try:
            calibration_data = {
                'target_width': self.target_width,
                'target_height': self.target_height,
                'camera_width': self.camera_width,
                'camera_height': self.camera_height,
                'calibration_points': self.calibration_points,
                'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
                'is_calibrated': self.is_calibrated
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
                
            print(f"Калибровка сохранена в файл: {filename}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения калибровки: {e}")
            return False
    
    def load_calibration(self, filename: str = None) -> bool:
        """
        Загрузка калибровки из файла
        
        Args:
            filename: Имя файла для загрузки
            
        Returns:
            True если загрузка прошла успешно
        """
        if filename is None:
            filename = self.calibration_file
            
        if not os.path.exists(filename):
            print(f"Файл калибровки не найден: {filename}")
            return False
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
                
            self.target_width = calibration_data.get('target_width', self.target_width)
            self.target_height = calibration_data.get('target_height', self.target_height)
            self.camera_width = calibration_data.get('camera_width', self.camera_width)
            self.camera_height = calibration_data.get('camera_height', self.camera_height)
            self.calibration_points = calibration_data.get('calibration_points', [])
            self.is_calibrated = calibration_data.get('is_calibrated', False)
            
            # Восстановление матрицы гомографии
            homography_data = calibration_data.get('homography_matrix')
            if homography_data is not None:
                self.homography_matrix = np.array(homography_data, dtype=np.float32)
                # Вычисление обратной матрицы
                try:
                    self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
                except np.linalg.LinAlgError:
                    print("Ошибка: не удалось вычислить обратную матрицу")
                    self.is_calibrated = False
            
            print(f"Калибровка загружена из файла: {filename}")
            print(f"Точек калибровки: {len(self.calibration_points)}")
            print(f"Статус калибровки: {'готова' if self.is_calibrated else 'не готова'}")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки калибровки: {e}")
            return False
    
    def draw_calibration_overlay(self, image: np.ndarray) -> np.ndarray:
        """
        Рисование наложения для калибровки
        
        Args:
            image: Исходное изображение
            
        Returns:
            Изображение с наложением
        """
        overlay = image.copy()
        
        # Рисование уже установленных точек
        for i, (x, y) in enumerate(self.calibration_points):
            color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
            cv2.circle(overlay, (x, y), 8, color, -1)
            cv2.putText(overlay, str(i+1), (x-10, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Если калибровка завершена, показываем упорядоченные точки
        if self.is_calibrated and len(self.calibration_points) == 4:
            # Рассчитываем упорядоченные точки для отображения
            points = self.calibration_points.copy()
            center_x = sum(p[0] for p in points) / len(points)
            center_y = sum(p[1] for p in points) / len(points)
            
            # Находим точки по углам
            corners = {"TL": None, "TR": None, "BL": None, "BR": None}
            corner_labels = []
            
            for x, y in points:
                if x < center_x and y < center_y:
                    corners["TL"] = (x, y)
                    corner_labels.append((x, y, "TL"))
                elif x >= center_x and y < center_y:
                    corners["TR"] = (x, y)
                    corner_labels.append((x, y, "TR"))
                elif x < center_x and y >= center_y:
                    corners["BL"] = (x, y)
                    corner_labels.append((x, y, "BL"))
                else:
                    corners["BR"] = (x, y)
                    corner_labels.append((x, y, "BR"))
            
            # Рисуем метки углов
            for x, y, label in corner_labels:
                cv2.putText(overlay, label, (x+15, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Если есть достаточно точек, рисуем контур области
        if len(self.calibration_points) >= 3:
            points = np.array(self.calibration_points, dtype=np.int32)
            if len(self.calibration_points) == 4:
                # Замкнутый контур
                cv2.polylines(overlay, [points], True, (0, 255, 0) if self.is_calibrated else (0, 255, 255), 2)
            else:
                # Незамкнутый контур
                cv2.polylines(overlay, [points], False, (255, 0, 0), 2)
        
        # Инструкции
        instructions = [
            f"Калибровка: {len(self.calibration_points)}/4 точек",
            "Кликните по 4 углам проекционной области",
            "ESC - выход, R - сброс, S - сохранить"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.is_calibrated:
            cv2.putText(overlay, "КАЛИБРОВКА ГОТОВА!", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return overlay


class InteractiveCalibrator:
    """Интерактивный калибровщик с интерфейсом мыши"""
    
    def __init__(self, calibration_manager: CalibrationManager) -> None:
        self.calibration_manager = calibration_manager
        self.current_image: Optional[np.ndarray] = None
        
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Обработчик событий мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.calibration_manager.is_calibrated:
                success = self.calibration_manager.add_calibration_point(x, y)
                if success and self.calibration_manager.is_calibrated:
                    print("Калибровка завершена! Нажмите 'S' для сохранения.")
    
    def run_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Запуск интерактивной калибровки
        
        Args:
            image: Изображение для калибровки
            
        Returns:
            Изображение с наложением калибровки
        """
        self.current_image = image
        return self.calibration_manager.draw_calibration_overlay(image)


def test_calibration() -> None:
    """Тестирование калибровки"""
    from realsense_camera import RealSenseManager
    
    camera = RealSenseManager()
    if camera.start():
        # Передаем реальное разрешение камеры
        calibration_manager = CalibrationManager(
            target_width=1920, target_height=1080,
            camera_width=camera.width, camera_height=camera.height
        )
    else:
        # Fallback к стандартным значениям
        calibration_manager = CalibrationManager()
    
    calibrator = InteractiveCalibrator(calibration_manager)
    
    # Попытка загрузить существующую калибровку
    calibration_manager.load_calibration()
    
    if not hasattr(camera, 'profile') or camera.profile is None:
        return
        
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', calibrator.mouse_callback)
    
    try:
        print("Калибровка проекции:")
        print("- Кликните мышью по 4 углам проекционной области")
        print("- R - сброс калибровки")
        print("- S - сохранить калибровку") 
        print("- ESC - выход")
        
        while True:
            depth_image, color_image = camera.get_frames()
            
            if color_image is None:
                continue
                
            # Создание изображения с наложением калибровки
            overlay_image = calibrator.run_calibration(color_image)
            
            cv2.imshow('Calibration', overlay_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):  # Сброс
                calibration_manager.reset_calibration()
                print("Калибровка сброшена")
            elif key == ord('s') or key == ord('S'):  # Сохранение
                calibration_manager.save_calibration()
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_calibration()
