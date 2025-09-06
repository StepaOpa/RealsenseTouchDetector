"""
Модуль для визуализации результатов отслеживания ног
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import math
from foot_tracker import FootPosition


class VisualizationManager:
    """Менеджер для визуализации результатов отслеживания"""
    
    def __init__(self, target_width: int = 1920, target_height: int = 1080) -> None:
        """
        Инициализация менеджера визуализации
        
        Args:
            target_width: Ширина целевого экрана
            target_height: Высота целевого экрана
        """
        self.target_width = target_width
        self.target_height = target_height
        
        # Настройки визуализации
        self.show_trails = True
        self.trail_length = 30
        self.show_grid = True
        self.show_zones = False
        
        # История позиций для следов
        self.position_trails: Dict[int, List[Tuple[int, int, float]]] = {}
        
        # Цвета для разных стоп
        self.foot_colors = [
            (255, 100, 100),  # Красный
            (100, 255, 100),  # Зеленый
            (100, 100, 255),  # Синий
            (255, 255, 100),  # Желтый
            (255, 100, 255),  # Магента
            (100, 255, 255),  # Циан
            (255, 150, 50),   # Оранжевый
            (150, 50, 255),   # Фиолетовый
        ]
        
    def create_camera_view(self, 
                          color_image: np.ndarray,
                          mask_image: np.ndarray,
                          contours: List[np.ndarray],
                          foot_positions: List[FootPosition],
                          calibration_points: List[Tuple[int, int]] = None,
                          is_calibrated: bool = False,
                          show_mask: bool = True,
                          show_info: bool = True) -> np.ndarray:
        """
        Создание вида с камеры с наложениями
        
        Args:
            color_image: Исходное цветное изображение
            mask_image: Маска объектов
            contours: Контуры стоп
            foot_positions: Позиции стоп
            calibration_points: Точки калибровки проекции (4 точки)
            is_calibrated: Статус калибровки (True - готова, False - в процессе)
            show_mask: Показывать ли маску
            show_info: Показывать ли информацию
            
        Returns:
            Визуализированное изображение
        """
        result = color_image.copy()
        
        # Наложение маски
        if show_mask and mask_image is not None:
            # Проверка совместимости размеров
            if mask_image.shape[:2] == result.shape[:2]:
                mask_colored = cv2.applyColorMap(mask_image, cv2.COLORMAP_HOT)
                result = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
            else:
                # Изменение размера маски под исходное изображение
                mask_resized = cv2.resize(mask_image, (result.shape[1], result.shape[0]))
                mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_HOT)
                result = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
        
        # Рисование контуров
        if contours:
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Рисование области калибровки проекции
        if calibration_points and len(calibration_points) == 4:
            points = np.array(calibration_points, dtype=np.int32)
            # Зеленый цвет если калибровка готова, желтый если в процессе
            color = (0, 255, 0) if is_calibrated else (0, 255, 255)
            thickness = 3 if is_calibrated else 2
            
            # Рисование полигона области калибровки
            cv2.polylines(result, [points], True, color, thickness)
            
            # Добавляем полупрозрачную заливку для лучшей видимости
            if is_calibrated:
                overlay = result.copy()
                cv2.fillPoly(overlay, [points], (0, 255, 0, 50))
                cv2.addWeighted(result, 0.95, overlay, 0.05, 0, result)
            
            # Подписи углов области (TL, TR, BR, BL)
            labels = ["TL", "TR", "BR", "BL"]
            for i, ((x, y), label) in enumerate(zip(calibration_points, labels)):
                # Рисуем кружок в углу
                cv2.circle(result, (x, y), 8, color, -1)
                cv2.circle(result, (x, y), 10, (255, 255, 255), 2)
                
                # Подпись угла
                cv2.putText(result, label, (x + 15, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(result, label, (x + 15, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Рисование позиций стоп
        self._draw_foot_positions_camera(result, foot_positions)
        
        # Информационная панель
        if show_info:
            self._draw_info_panel(result, foot_positions)
        
        return result
    
    def create_screen_view(self, foot_positions: List[FootPosition]) -> np.ndarray:
        """
        Создание вида экранного пространства
        
        Args:
            foot_positions: Позиции стоп
            
        Returns:
            Изображение экранного пространства
        """
        # Создаем черный фон
        screen_view = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Рисование сетки
        if self.show_grid:
            self._draw_grid(screen_view)
        
        # Рисование зон
        if self.show_zones:
            self._draw_zones(screen_view)
        
        # Обновление следов
        self._update_trails(foot_positions)
        
        # Рисование следов
        if self.show_trails:
            self._draw_trails(screen_view)
        
        # Рисование текущих позиций стоп
        self._draw_foot_positions_screen(screen_view, foot_positions)
        
        # Заголовок
        cv2.putText(screen_view, f"Screen Space ({self.target_width}x{self.target_height})", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return screen_view
    
    def _draw_foot_positions_camera(self, image: np.ndarray, foot_positions: List[FootPosition]) -> None:
        """Рисование позиций стоп в виде камеры"""
        for i, foot in enumerate(foot_positions):
            # Выбор цвета
            color = self.foot_colors[i % len(self.foot_colors)]
            
            # Размер кружка зависит от уверенности
            radius = int(8 + 10 * foot.confidence)
            
            # Центр стопы
            cv2.circle(image, (foot.camera_x, foot.camera_y), radius, color, -1)
            cv2.circle(image, (foot.camera_x, foot.camera_y), radius + 2, (255, 255, 255), 2)
            
            # ID стопы
            cv2.putText(image, str(foot.id), 
                       (foot.camera_x - 8, foot.camera_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Информация о стопе
            info_lines = [
                f"ID: {foot.id}",
                f"Conf: {foot.confidence:.2f}",
                f"Screen: ({foot.screen_x}, {foot.screen_y})",
                f"Depth: {foot.depth:.2f}m"
            ]
            
            for j, line in enumerate(info_lines):
                cv2.putText(image, line,
                           (foot.camera_x + 20, foot.camera_y - 30 + j * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_foot_positions_screen(self, screen_view: np.ndarray, foot_positions: List[FootPosition]) -> None:
        """Рисование позиций стоп в экранном пространстве"""
        for i, foot in enumerate(foot_positions):
            # Выбор цвета
            color = self.foot_colors[i % len(self.foot_colors)]
            
            # Размер зависит от уверенности и площади
            base_radius = 30
            confidence_scale = 0.5 + 0.5 * foot.confidence
            area_scale = max(0.5, min(2.0, math.sqrt(foot.area / 1000)))
            radius = int(base_radius * confidence_scale * area_scale)
            
            # Главный кружок стопы
            cv2.circle(screen_view, (foot.screen_x, foot.screen_y), radius, color, -1)
            cv2.circle(screen_view, (foot.screen_x, foot.screen_y), radius + 3, (255, 255, 255), 2)
            
            # Пульсирующий эффект для активных стоп
            pulse_radius = int(radius + 10 * math.sin(time.time() * 3 + foot.id))
            if pulse_radius > radius:
                cv2.circle(screen_view, (foot.screen_x, foot.screen_y), pulse_radius, color, 2)
            
            # ID и информация
            cv2.putText(screen_view, str(foot.id),
                       (foot.screen_x - 10, foot.screen_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            
            # Координаты под стопой
            coord_text = f"({foot.screen_x}, {foot.screen_y})"
            cv2.putText(screen_view, coord_text,
                       (foot.screen_x - 50, foot.screen_y + radius + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_grid(self, screen_view: np.ndarray) -> None:
        """Рисование координатной сетки"""
        grid_color = (40, 40, 40)
        
        # Вертикальные линии
        for x in range(0, self.target_width, 100):
            cv2.line(screen_view, (x, 0), (x, self.target_height), grid_color, 1)
            if x % 500 == 0:  # Жирные линии каждые 500 пикселей
                cv2.line(screen_view, (x, 0), (x, self.target_height), (80, 80, 80), 2)
                cv2.putText(screen_view, str(x), (x + 5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        # Горизонтальные линии
        for y in range(0, self.target_height, 100):
            cv2.line(screen_view, (0, y), (self.target_width, y), grid_color, 1)
            if y % 500 == 0:  # Жирные линии каждые 500 пикселей
                cv2.line(screen_view, (0, y), (self.target_width, y), (80, 80, 80), 2)
                if y > 0:
                    cv2.putText(screen_view, str(y), (5, y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    
    def _draw_zones(self, screen_view: np.ndarray) -> None:
        """Рисование интерактивных зон"""
        # Пример зон (можно настроить)
        zones = [
            {"name": "Zone 1", "rect": (100, 100, 400, 300), "color": (50, 50, 100)},
            {"name": "Zone 2", "rect": (600, 200, 300, 400), "color": (50, 100, 50)},
            {"name": "Zone 3", "rect": (1200, 150, 500, 250), "color": (100, 50, 50)},
        ]
        
        for zone in zones:
            x, y, w, h = zone["rect"]
            color = zone["color"]
            
            # Полупрозрачный прямоугольник
            overlay = screen_view.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(screen_view, 0.8, overlay, 0.2, 0, screen_view)
            
            # Граница зоны
            cv2.rectangle(screen_view, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Название зоны
            cv2.putText(screen_view, zone["name"], (x + 10, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _update_trails(self, foot_positions: List[FootPosition]) -> None:
        """Обновление следов движения стоп"""
        current_time = time.time()
        
        # Добавление новых позиций в следы
        for foot in foot_positions:
            if foot.id not in self.position_trails:
                self.position_trails[foot.id] = []
            
            # Добавляем новую позицию
            self.position_trails[foot.id].append((foot.screen_x, foot.screen_y, current_time))
            
            # Ограничиваем длину следа
            if len(self.position_trails[foot.id]) > self.trail_length:
                self.position_trails[foot.id].pop(0)
        
        # Удаление старых следов (стопы, которые исчезли)
        active_ids = {foot.id for foot in foot_positions}
        expired_ids = []
        
        for foot_id in self.position_trails:
            if foot_id not in active_ids:
                # Удаляем старые точки
                trail = self.position_trails[foot_id]
                trail[:] = [(x, y, t) for x, y, t in trail if current_time - t < 5.0]
                
                if not trail:
                    expired_ids.append(foot_id)
        
        # Удаляем пустые следы
        for foot_id in expired_ids:
            del self.position_trails[foot_id]
    
    def _draw_trails(self, screen_view: np.ndarray) -> None:
        """Рисование следов движения"""
        current_time = time.time()
        
        for foot_id, trail in self.position_trails.items():
            if len(trail) < 2:
                continue
                
            # Выбор цвета для следа
            color_idx = (foot_id - 1) % len(self.foot_colors)
            base_color = self.foot_colors[color_idx]
            
            # Рисование линий между точками следа
            for i in range(len(trail) - 1):
                x1, y1, t1 = trail[i]
                x2, y2, t2 = trail[i + 1]
                
                # Прозрачность зависит от возраста точки
                age_factor = max(0.1, 1.0 - (current_time - t2) / 5.0)
                alpha_color = tuple(int(c * age_factor) for c in base_color)
                
                # Толщина линии уменьшается со временем
                thickness = max(1, int(5 * age_factor))
                
                cv2.line(screen_view, (x1, y1), (x2, y2), alpha_color, thickness)
                
            # Рисование точек следа
            for x, y, t in trail[:-1]:  # Исключаем последнюю (текущую) точку
                age_factor = max(0.1, 1.0 - (current_time - t) / 5.0)
                radius = max(2, int(6 * age_factor))
                alpha_color = tuple(int(c * age_factor) for c in base_color)
                cv2.circle(screen_view, (x, y), radius, alpha_color, -1)
    
    def _draw_info_panel(self, image: np.ndarray, foot_positions: List[FootPosition]) -> None:
        """Рисование информационной панели"""
        panel_height = 120
        panel_color = (0, 0, 0)
        
        # Создание полупрозрачной панели
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), panel_color, -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Заголовок
        cv2.putText(image, "Foot Tracking System - Camera View", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Статистика
        info_text = [
            f"Feet detected: {len(foot_positions)}",
            f"Active trails: {len(self.position_trails)}",
            f"Target screen: {self.target_width}x{self.target_height}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (10, 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Управление
        controls = [
            "Controls: T-trails, G-grid, Z-zones, ESC-exit"
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(image, text, (400, 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
    
    def toggle_trails(self) -> None:
        """Переключение отображения следов"""
        self.show_trails = not self.show_trails
        print(f"Отображение следов: {'включено' if self.show_trails else 'выключено'}")
    
    def toggle_grid(self) -> None:
        """Переключение отображения сетки"""
        self.show_grid = not self.show_grid
        print(f"Отображение сетки: {'включено' if self.show_grid else 'выключено'}")
    
    def toggle_zones(self) -> None:
        """Переключение отображения зон"""
        self.show_zones = not self.show_zones
        print(f"Отображение зон: {'включено' if self.show_zones else 'выключено'}")
    
    def clear_trails(self) -> None:
        """Очистка всех следов"""
        self.position_trails.clear()
        print("Следы очищены")


def test_visualization() -> None:
    """Тестирование модуля визуализации"""
    from foot_tracker import FootTracker
    
    tracker = FootTracker()
    visualizer = VisualizationManager()
    
    if not tracker.initialize():
        return
    
    try:
        print("Визуализация запущена!")
        print("Управление:")
        print("  T - переключить следы")
        print("  G - переключить сетку")
        print("  Z - переключить зоны")
        print("  C - очистить следы")
        print("  ESC - выход")
        
        while True:
            debug_image, mask_image, foot_positions = tracker.process_frame()
            
            if debug_image is None:
                continue
            
            # Создание видов
            camera_view = visualizer.create_camera_view(
                debug_image, mask_image, [], foot_positions,
                calibration_points=[], is_calibrated=False
            )
            screen_view = visualizer.create_screen_view(foot_positions)
            
            # Отображение
            cv2.imshow('Camera View', camera_view)
            cv2.imshow('Screen Space', screen_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('t') or key == ord('T'):
                visualizer.toggle_trails()
            elif key == ord('g') or key == ord('G'):
                visualizer.toggle_grid()
            elif key == ord('z') or key == ord('Z'):
                visualizer.toggle_zones()
            elif key == ord('c') or key == ord('C'):
                visualizer.clear_trails()
                
    finally:
        tracker.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_visualization()
