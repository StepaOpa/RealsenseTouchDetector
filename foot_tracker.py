"""
Основной класс для отслеживания ног и вычисления координат
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass

from realsense_camera import RealSenseManager
from floor_detector import FloorDetector
from calibration import CalibrationManager


@dataclass
class FootPosition:
    """Структура для хранения информации о позиции стопы"""
    id: int
    camera_x: int
    camera_y: int
    screen_x: int
    screen_y: int
    depth: float
    world_x: float
    world_y: float
    world_z: float
    confidence: float
    timestamp: float
    area: float


class FootTracker:
    """Главный класс для отслеживания ног"""
    
    def __init__(self, 
                 target_width: int = 1920, 
                 target_height: int = 1080,
                 floor_depth: float = 1.5,
                 floor_tolerance: float = 0.2) -> None:
        """
        Инициализация трекера ног
        
        Args:
            target_width: Целевая ширина экрана
            target_height: Целевая высота экрана
            floor_depth: Глубина пола в метрах
            floor_tolerance: Допуск по глубине в метрах
        """
        # Инициализация компонентов
        self.camera: RealSenseManager = RealSenseManager()
        self.floor_detector: FloorDetector = FloorDetector(floor_depth, floor_tolerance)
        self.calibration: CalibrationManager = CalibrationManager(target_width, target_height)
        
        # Параметры трекинга
        self.max_feet_count: int = 10
        self.min_confidence: float = 0.3
        self.tracking_history_size: int = 10
        
        # История позиций для сглаживания
        self.position_history: List[List[FootPosition]] = []
        
        # Счетчик ID для стоп
        self.next_foot_id: int = 1
        
        # Статистика
        self.frame_count: int = 0
        self.fps: float = 0.0
        
        self.last_fps_time: float = time.time()
        
        # Текущие позиции стоп
        self.current_foot_positions: List[FootPosition] = []
        
        # Отфильтрованные позиции (вне области проекции)
        self.filtered_positions: List[Tuple[int, int]] = []
        
    def initialize(self, load_calibration: bool = True) -> bool:
        """
        Инициализация системы
        
        Args:
            load_calibration: Загружать ли существующую калибровку
            
        Returns:
            True если инициализация прошла успешно
        """
        print("Инициализация системы отслеживания ног...")
        
        # Запуск камеры
        if not self.camera.start():
            print("Ошибка: не удалось запустить камеру")
            return False
            
        # Проверка и отображение размеров камеры
        print(f"📹 Размер камеры: {self.camera.width}x{self.camera.height}")
        
        # Установка реального разрешения камеры в калибровке
        self.calibration.set_camera_resolution(self.camera.width, self.camera.height)
        
        print(f"🎯 Калибратор - размер камеры: {self.calibration.camera_width}x{self.calibration.camera_height}")
        print(f"🖥️ Калибратор - размер экрана: {self.calibration.target_width}x{self.calibration.target_height}")
        
        # Тестируем преобразование координат центра камеры
        center_x = self.camera.width // 2
        center_y = self.camera.height // 2
        test_screen_x, test_screen_y = self.calibration.transform_point(center_x, center_y, debug=False)
        expected_screen_x = self.calibration.target_width // 2
        expected_screen_y = self.calibration.target_height // 2
        
        print(f"🧪 ТЕСТ КООРДИНАТ: центр камеры({center_x}, {center_y}) -> экран({test_screen_x}, {test_screen_y})")
        print(f"   Ожидаемый центр экрана: ({expected_screen_x}, {expected_screen_y})")
        
        # Проверяем, работает ли преобразование координат правильно
        if abs(test_screen_x - expected_screen_x) < 50 and abs(test_screen_y - expected_screen_y) < 50:
            print("✅ Преобразование координат работает правильно!")
        else:
            print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Преобразование координат может работать неточно!")
            
        # Загрузка калибровки
        if load_calibration:
            self.calibration.load_calibration()
            
        print("Система инициализирована успешно!")
        return True
        
    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[FootPosition]]:
        """
        Обработка одного кадра
        
        Returns:
            Tuple с debug_image, mask_image и списком позиций стоп
        """
        # Получение кадров с камеры
        depth_image, color_image = self.camera.get_frames()
        
        if depth_image is None or color_image is None:
            return None, None, []
            
        # Детекция объектов на полу
        objects_mask, filtered_mask = self.floor_detector.detect_objects_on_floor(
            depth_image, self.camera.depth_scale
        )
        
        # Поиск контуров стоп
        contours = self.floor_detector.find_foot_contours(filtered_mask)
        centers = self.floor_detector.get_contour_centers(contours)
        
        # Обработка найденных стоп
        foot_positions = self._process_detected_feet(
            centers, contours, depth_image, color_image
        )
        
        # Обновление истории
        self._update_tracking_history(foot_positions)
        
        # Создание debug изображения
        debug_image = self._create_debug_visualization(
            color_image, filtered_mask, contours, foot_positions
        )
        
        # Обновление статистики
        self._update_statistics()
        
        self.current_foot_positions = foot_positions
        return debug_image, filtered_mask, foot_positions
        
    def _process_detected_feet(self, 
                              centers: List[Tuple[int, int]], 
                              contours: List[np.ndarray],
                              depth_image: np.ndarray,
                              color_image: np.ndarray) -> List[FootPosition]:
        """Обработка обнаруженных стоп"""
        foot_positions = []
        
        # Статистика фильтрации
        total_detected = len(centers)
        filtered_out = 0
        filtered_positions = []  # Для визуализации отфильтрованных стоп
        
        for i, ((cx, cy), contour) in enumerate(zip(centers, contours)):
            # ФИЛЬТРАЦИЯ: Проверяем, находится ли стопа в области проекции
            if not self.calibration.is_point_in_projection_area(cx, cy):
                filtered_out += 1
                # Сохраняем для визуализации
                filtered_positions.append((cx, cy))
                continue
                
            # Получение глубины в центре стопы
            depth = self.camera.get_depth_at_pixel(cx, cy, depth_image)
            
            # Пропуск если глубина некорректна
            if depth <= 0:
                continue
                
            # Преобразование в 3D координаты
            world_x, world_y, world_z = self.camera.pixel_to_3d(cx, cy, depth)
            
            # Преобразование в экранные координаты
            screen_x, screen_y = self.calibration.transform_point(cx, cy, debug=False)
            
            # Отладочный вывод для первого касания (отключен)
            # if is_first_touch:
            #     mode = "Гомография" if self.calibration.is_calibrated else "Масштабирование"
            #     print(f"🎯 Касание: камера({cx}, {cy}) -> экран({screen_x}, {screen_y}) [{mode}]")
            
            # Вычисление уверенности на основе площади и формы контура
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            confidence = self._calculate_confidence(area, perimeter, depth)
            
            # Создание объекта позиции стопы
            foot_position = FootPosition(
                id=self.next_foot_id,
                camera_x=cx,
                camera_y=cy,
                screen_x=screen_x,
                screen_y=screen_y,
                depth=depth,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                confidence=confidence,
                timestamp=time.time(),
                area=area
            )
            
            # Добавляем только если уверенность достаточна
            if confidence >= self.min_confidence:
                foot_positions.append(foot_position)
                self.next_foot_id += 1
        
        # Сохраняем отфильтрованные позиции для визуализации
        self.filtered_positions = filtered_positions
        
        # Вывод статистики фильтрации (только если есть активность)
        if total_detected > 0:
            accepted = len(foot_positions)
            if filtered_out > 0:
                print(f"\n📍 ФИЛЬТРАЦИЯ ПО ОБЛАСТИ: Обнаружено: {total_detected}, Принято: {accepted}, Отфильтровано: {filtered_out}")
                
        return foot_positions
        
    def _calculate_confidence(self, area: float, perimeter: float, depth: float) -> float:
        """
        Вычисление уверенности детекции
        
        Args:
            area: Площадь контура
            perimeter: Периметр контура  
            depth: Глубина
            
        Returns:
            Уверенность от 0.0 до 1.0
        """
        confidence = 1.0
        
        # Штраф за слишком маленькую или большую площадь
        if area < 500:
            confidence *= (area / 500)
        elif area > 20000:
            confidence *= max(0.1, (30000 - area) / 10000)
            
        # Штраф за неправильную форму (слишком вытянутые объекты)
        if perimeter > 0:
            compactness = (perimeter * perimeter) / area
            if compactness > 50:
                confidence *= max(0.1, (100 - compactness) / 50)
                
        # Штраф за некорректную глубину
        if depth <= 0 or depth > 5.0:  # Больше 5 метров - подозрительно
            confidence *= 0.1
            
        return max(0.0, min(1.0, confidence))
        
    def _update_tracking_history(self, foot_positions: List[FootPosition]) -> None:
        """Обновление истории отслеживания"""
        self.position_history.append(foot_positions.copy())
        
        # Ограничение размера истории
        if len(self.position_history) > self.tracking_history_size:
            self.position_history.pop(0)
            
    def _create_debug_visualization(self, 
                                   color_image: np.ndarray,
                                   mask: np.ndarray,
                                   contours: List[np.ndarray],
                                   foot_positions: List[FootPosition]) -> np.ndarray:
        """Создание отладочной визуализации"""
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
        
        # Рисование позиций стоп
        for foot in foot_positions:
            # Цвет зависит от уверенности
            confidence_color = int(255 * foot.confidence)
            color = (0, confidence_color, 255 - confidence_color)
            
            # Центр стопы
            cv2.circle(debug_image, (foot.camera_x, foot.camera_y), 8, color, -1)
            
            # Информация о стопе
            info_text = f"ID:{foot.id} C:{foot.confidence:.2f}"
            cv2.putText(debug_image, info_text, 
                       (foot.camera_x - 30, foot.camera_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Координаты
            coord_text = f"({foot.screen_x}, {foot.screen_y})"
            cv2.putText(debug_image, coord_text,
                       (foot.camera_x - 40, foot.camera_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
            # Глубина
            depth_text = f"{foot.depth:.2f}m"
            cv2.putText(debug_image, depth_text,
                       (foot.camera_x - 20, foot.camera_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Рисование отфильтрованных стоп (вне области проекции)
        for x, y in self.filtered_positions:
            # Красный кружок с крестиком для отфильтрованных стоп
            cv2.circle(debug_image, (x, y), 10, (0, 0, 255), 2)  # Красный круг
            cv2.line(debug_image, (x-8, y-8), (x+8, y+8), (0, 0, 255), 2)  # Крестик \
            cv2.line(debug_image, (x-8, y+8), (x+8, y-8), (0, 0, 255), 2)  # Крестик /
            
            # Подпись "FILTERED"
            cv2.putText(debug_image, "FILTERED", 
                       (x - 25, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Статистика
        stats_text = [
            f"FPS: {self.fps:.1f}",
            f"Кадр: {self.frame_count}",
            f"Стоп: {len(foot_positions)}",
            f"Отфильтровано: {len(self.filtered_positions)}",
            f"Пол: {self.floor_detector.floor_depth:.2f}м",
            f"Калибровка: {'ОК' if self.calibration.is_calibrated else 'НЕТ'}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(debug_image, text, (10, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                       
        return debug_image
        
    def _update_statistics(self) -> None:
        """Обновление статистики"""
        self.frame_count += 1
        
        # Вычисление FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
    def get_smoothed_positions(self, window_size: int = 5) -> List[FootPosition]:
        """
        Получение сглаженных позиций стоп
        
        Args:
            window_size: Размер окна для сглаживания
            
        Returns:
            Список сглаженных позиций
        """
        if len(self.position_history) < window_size:
            return self.current_foot_positions
            
        # Пока возвращаем текущие позиции
        # В будущем здесь можно реализовать сглаживание по истории
        return self.current_foot_positions
        
    def calibrate_floor_depth(self) -> float:
        """
        Автоматическая калибровка глубины пола
        
        Returns:
            Определенная глубина пола
        """
        depth_image, _ = self.camera.get_frames()
        if depth_image is None:
            return self.floor_detector.floor_depth
            
        floor_depth = self.floor_detector.auto_detect_floor(depth_image, self.camera.depth_scale)
        print(f"Глубина пола автоматически установлена: {floor_depth:.3f} м")
        return floor_depth
        
    def set_floor_depth(self, depth: float) -> None:
        """Установка глубины пола вручную"""
        self.floor_detector.set_floor_depth(depth)
        
    def get_camera_info(self) -> Dict:
        """Получение информации о камере"""
        return self.camera.get_camera_info()
        
    def cleanup(self) -> None:
        """Очистка ресурсов"""
        self.camera.stop()
        
    def __del__(self) -> None:
        """Деструктор"""
        self.cleanup()


def test_foot_tracker() -> None:
    """Тестирование трекера ног"""
    tracker = FootTracker()
    
    if not tracker.initialize():
        return
        
    try:
        print("Трекер ног запущен!")
        print("Управление:")
        print("  C - автокалибровка глубины пола")
        print("  + - увеличить глубину пола на 10см")
        print("  - - уменьшить глубину пола на 10см") 
        print("  S - сохранить калибровку")
        print("  ESC - выход")
        
        while True:
            debug_image, mask_image, foot_positions = tracker.process_frame()
            
            if debug_image is None:
                continue
                
            # Отображение результатов
            cv2.imshow('Foot Tracker - Debug', debug_image)
            cv2.imshow('Foot Tracker - Mask', mask_image)
            
            # Вывод координат в консоль
            if foot_positions:
                print(f"\rСтоп обнаружено: {len(foot_positions)} ", end="")
                for foot in foot_positions:
                    print(f"[{foot.screen_x}, {foot.screen_y}] ", end="")
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c') or key == ord('C'):
                tracker.calibrate_floor_depth()
            elif key == ord('+') or key == ord('='):
                new_depth = tracker.floor_detector.floor_depth + 0.1
                tracker.set_floor_depth(new_depth)
                print(f"\nГлубина пола: {new_depth:.2f}м")
            elif key == ord('-'):
                new_depth = tracker.floor_detector.floor_depth - 0.1
                tracker.set_floor_depth(max(0.5, new_depth))
                print(f"\nГлубина пола: {new_depth:.2f}м")
            elif key == ord('s') or key == ord('S'):
                tracker.calibration.save_calibration()
                
    finally:
        tracker.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_foot_tracker()
