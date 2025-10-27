"""
Модуль для обработки касаний на кропнутом потоке изображений
Определяет касания пользователя и масштабирует координаты до целевого разрешения
"""

import cv2
import numpy as np
from numpy.linalg import lstsq
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
from touch_filter import TouchFilter


class TouchPoint:
    """
    Класс для представления точки касания
    """

    def __init__(
        self, x: int, y: int, depth: float, area: int, confidence: float = 1.0
    ) -> None:
        """
        Инициализация точки касания

        Args:
            x: X координата в кропнутом изображении
            y: Y координата в кропнутом изображении
            depth: Глубина касания в метрах
            area: Площадь области касания в пикселях
            confidence: Уверенность в касании (0.0-1.0)
        """
        self.x: int = x
        self.y: int = y
        self.depth: float = depth
        self.area: int = area
        self.confidence: float = confidence
        self.timestamp: float = time.time()
        # Смещение координат касания (в пикселях)


class TouchProcessor:
    """
    Класс для обработки касаний на кропнутом потоке
    """

    def __init__(
        self,
        cropped_width: int,
        cropped_height: int,
        target_width: int = 1920,
        target_height: int = 1080,
    ) -> None:
        """
        Инициализация процессора касаний

        Args:
            cropped_width: Ширина кропнутого изображения
            cropped_height: Высота кропнутого изображения
            target_width: Целевая ширина для масштабирования (по умолчанию 1920)
            target_height: Целевая высота для масштабирования (по умолчанию 1080)
        """
        self.cropped_width: int = cropped_width
        self.cropped_height: int = cropped_height
        self.target_width: int = target_width
        self.target_height: int = target_height
        self.plane_A: Optional[float] = None
        self.plane_B: Optional[float] = None
        self.plane_C: Optional[float] = None
        self.plane_rmse: float = 0.0
        self.plane_max_error: float = 0.0

        self.touch_offset_x: int = 0
        self.touch_offset_y: int = 0

        # Для визуализации
        self.last_surface_diff: Optional[np.ndarray] = None

        # Фильтрация по глубине - исключение объектов выше определенного уровня
        self.min_touch_depth: float = (
            0.3  # Минимальная глубина касания в метрах (30 см)
        )
        self.max_touch_depth: float = (
            2.0  # Максимальная глубина касания в метрах (2 метра)
        )
        self.enable_depth_filtering: bool = (
            True  # Включить/выключить фильтрацию по глубине
        )

        # Коэффициенты масштабирования
        self.scale_x: float = target_width / cropped_width
        self.scale_y: float = target_height / cropped_height

        # Настройка логирования
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Параметры детекции касаний
        self.background_depth: Optional[np.ndarray] = None
        self.touch_threshold: float = 0.015  # Порог касания в метрах (1.5 см)
        self.background_threshold: float = 0.015  # Порог фона в метрах (1.5 см)
        self.min_touch_area: int = 500  # Минимальная площадь касания в пикселях
        self.max_touch_area: int = 150000  # Максимальная площадь касания в пикселях
        self.background_update_rate: float = 0.02  # Скорость обновления фона (0.0-1.0)

        # Калибровка плоскости проекции
        self.surface_height: Optional[float] = (
            None  # Средняя высота поверхности проекции
        )
        self.surface_tolerance: float = 0.2  # Допуск для определения поверхности (5 см)
        self.surface_calibration_points: List[Tuple[int, int, float]] = (
            []
        )  # Точки калибровки поверхности

        # Настройки чувствительности
        self.sensitivity_level: int = 5  # Уровень чувствительности (1-10)
        self.depth_noise_threshold: float = 0.01  # Порог шума глубины (5 мм)

        # Фильтрация и сглаживание
        self.noise_filter_size: int = 3  # Размер морфологического фильтра
        self.gaussian_blur_size: int = 5  # Размер гауссова размытия

        # Коррекция глубины для устранения параллакса камера-проектор
        self.depth_offset_mm: float = (
            0.0  # Смещение глубины в миллиметрах (-100 до +200мм)
        )
        self.depth_scale_factor: float = (
            1.0  # Коэффициент масштабирования глубины (0.8 до 1.5)
        )
        self.enable_spatial_filter: bool = (
            True  # Включить/выключить медианную фильтрацию
        )
        self.spatial_filter_kernel: int = (
            5  # Размер ядра медианного фильтра (3, 5, 7, 9)
        )
        self.max_depth_mm: int = 10000  # Максимальная глубина в мм (10 метров)

        # История касаний для стабилизации
        self.touch_history: List[List[TouchPoint]] = []
        self.history_length: int = 3

        # Продвинутый фильтр касаний
        self.touch_filter: TouchFilter = TouchFilter(
            distance_threshold=30.0,
            history_size=5,
            min_movement_threshold=10.0,
            timeout=2.0,
        )
        self.use_advanced_filter: bool = (
            True  # Включение/выключение продвинутой фильтрации
        )

        # Статистика
        self.frame_count: int = 0
        self.total_touches_detected: int = 0

        # Отладочные изображения
        self.last_binary_image: Optional[np.ndarray] = (
            None  # Последнее бинарное изображение
        )
        self.last_depth_diff: Optional[np.ndarray] = (
            None  # Последнее разностное изображение
        )

        self.logger.info(
            f"TouchProcessor инициализирован: {cropped_width}x{cropped_height} -> {target_width}x{target_height}"
        )
        self.logger.info(
            f"Коэффициенты масштабирования: X={self.scale_x:.3f}, Y={self.scale_y:.3f}"
        )
        self.logger.info(
            f"Начальная чувствительность: {self.sensitivity_level}/10, порог касания: {self.touch_threshold:.3f}м"
        )

    def get_min_touch_area(self) -> int:
        return self.min_touch_area

    def set_min_touch_area(self, value: int) -> None:
        self.min_touch_area = max(10, value)  # Минимум 10 пикселей
        self.logger.debug(f"Минимальная площадь касания: {self.min_touch_area}")

    def get_max_touch_area(self) -> int:
        return self.max_touch_area

    def set_max_touch_area(self, value: int) -> None:
        self.max_touch_area = max(self.min_touch_area, value)
        self.logger.debug(f"Максимальная площадь касания: {self.max_touch_area}")

    def _apply_depth_filter(
        self, depth_image: np.ndarray, binary_image: np.ndarray
    ) -> np.ndarray:
        """
        Применяет фильтрацию по глубине к бинарному изображению

        Args:
            depth_image: Изображение глубины в миллиметрах
            binary_image: Бинарное изображение касаний

        Returns:
            Отфильтрованное бинарное изображение
        """
        if not self.enable_depth_filtering:
            return binary_image

        # Создаем маску допустимой глубины
        depth_meters = depth_image.astype(np.float32) * 0.001  # Конвертируем в метры

        # Маска для допустимого диапазона глубины
        valid_depth_mask = (depth_meters >= self.min_touch_depth) & (
            depth_meters <= self.max_touch_depth
        )

        # Применяем маску к бинарному изображению
        filtered_binary = binary_image.copy()
        filtered_binary[~valid_depth_mask] = 0

        # Морфологическая обработка для устранения шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered_binary = cv2.morphologyEx(filtered_binary, cv2.MORPH_OPEN, kernel)

        return filtered_binary

    def set_background(self, depth_image: np.ndarray) -> None:
        """
        Установка фонового изображения глубины

        Args:
            depth_image: Изображение глубины для установки как фон
        """
        if depth_image.shape != (self.cropped_height, self.cropped_width):
            self.logger.error(
                f"Размер изображения глубины {depth_image.shape} не соответствует ожидаемому {(self.cropped_height, self.cropped_width)}"
            )
            return

        self.background_depth = depth_image.astype(np.float32).copy()
        self.logger.info("Фоновое изображение глубины установлено")

    def update_background(self, depth_image: np.ndarray) -> None:
        """
        Постепенное обновление фонового изображения

        Args:
            depth_image: Текущее изображение глубины
        """

        if self.background_depth is None:
            self.set_background(depth_image)
            return

        # Вычисляем разность с текущим фоном
        current_diff = np.abs(depth_image.astype(np.float32) - self.background_depth)
        motion_mask = current_diff < (self.touch_threshold * 1000)  # Порог в мм

        # Обновляем только статические области
        depth_float = depth_image.astype(np.float32)
        self.background_depth = np.where(
            motion_mask,
            (1 - self.background_update_rate) * self.background_depth
            + self.background_update_rate * depth_float,
            self.background_depth,
        )
        # if self.background_depth is None:
        #     self.set_background(depth_image)
        #     return

        # # Постепенное обновление фона с учетом коэффициента обновления
        # depth_float = depth_image.astype(np.float32)
        # self.background_depth = (1 - self.background_update_rate) * self.background_depth + self.background_update_rate * depth_float

    def add_surface_calibration_point(
        self, x: int, y: int, depth_image: np.ndarray
    ) -> bool:
        """
        Добавление точки для калибровки поверхности проекции

        Args:
            x: X координата точки
            y: Y координата точки
            depth_image: Изображение глубины для получения значения

        Returns:
            True если точка добавлена успешно
        """
        if 0 <= x < self.cropped_width and 0 <= y < self.cropped_height:
            # Получаем значение глубины в точке с усреднением по окрестности
            region_size = 5
            x1, y1 = max(0, x - region_size), max(0, y - region_size)
            x2, y2 = min(self.cropped_width, x + region_size), min(
                self.cropped_height, y + region_size
            )

            region = depth_image[y1:y2, x1:x2]
            valid_depths = region[region > 0]

            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths) * 0.001  # Конвертируем в метры
                self.surface_calibration_points.append((x, y, avg_depth))
                self.logger.info(
                    f"Точка калибровки добавлена: ({x}, {y}) глубина={avg_depth:.3f}м"
                )
                return True

        self.logger.warning(f"Не удалось добавить точку калибровки в ({x}, {y})")
        return False

    def calibrate_surface_height(self) -> bool:
        """
        Калибровка поверхности с помощью регрессии плоскости
        z = Ax + By + C
        """
        if len(self.surface_calibration_points) < 3:
            self.logger.warning(
                "Недостаточно точек для регрессии плоскости (минимум 3)"
            )
            return False

        points = np.array(self.surface_calibration_points)
        x_coords = points[:, 0]  # X пиксели
        y_coords = points[:, 1]  # Y пиксели
        depths = points[:, 2]  # Глубина в метрах

        # Матрица для уравнения плоскости
        A_matrix = np.column_stack([x_coords, y_coords, np.ones(len(x_coords))])

        try:
            # Решаем методом наименьших квадратов
            coefficients, residuals, rank, s = lstsq(A_matrix, depths)
            self.plane_A, self.plane_B, self.plane_C = coefficients

            # Вычисляем ошибку аппроксимации
            predicted_depths = A_matrix @ coefficients
            errors = np.abs(depths - predicted_depths)
            self.plane_rmse = np.sqrt(np.mean(errors**2))
            self.plane_max_error = np.max(errors)

            self.logger.info(
                f"Плоскость калибрована: z = {self.plane_A:.6f}*x + {self.plane_B:.6f}*y + {self.plane_C:.3f}"
            )
            self.logger.info(
                f"Точность: RMSE={self.plane_rmse:.4f}м, MaxError={self.plane_max_error:.4f}м"
            )

            # Предупреждение если точность низкая
            if self.plane_rmse > 0.02:
                self.logger.warning(
                    "Низкая точность калибровки! Рекомендуется добавить больше точек."
                )

            return True

        except Exception as e:
            self.logger.error(f"Ошибка регрессии плоскости: {e}")
            return False

    def clear_surface_calibration(self) -> None:
        """Очистка точек калибровки и плоскости"""
        self.surface_calibration_points.clear()
        self.surface_height = None
        self.plane_A = None
        self.plane_B = None
        self.plane_C = None
        self.logger.info("Калибровка поверхности сброшена")

    def adjust_sensitivity(self, delta: int) -> None:
        """
        Настройка чувствительности детекции касаний

        Args:
            delta: Изменение уровня чувствительности (+1 или -1)
        """
        self.sensitivity_level = max(1, min(10, self.sensitivity_level + delta))

        # Пересчитываем параметры детекции на основе уровня чувствительности
        base_threshold = 0.010  # Базовый порог м
        base_min_area = 1500  # Базовая минимальная площадь
        base_noise_threshold = 0.2  # Базовый порог шума

        # Чем выше чувствительность, тем ниже пороги
        sensitivity_factor = (11 - self.sensitivity_level) / 10.0  # От 1.0 до 0.1

        self.touch_threshold = base_threshold * sensitivity_factor
        self.min_touch_area = int(base_min_area * sensitivity_factor)
        self.depth_noise_threshold = base_noise_threshold * sensitivity_factor

        # Обновляем скорость обновления фона (более чувствительные настройки = медленнее обновление)
        self.background_update_rate = 0.01 + (self.sensitivity_level - 1) * 0.008

        self.logger.info(f"Чувствительность: {self.sensitivity_level}/10")
        self.logger.info(f"  Порог касания: {self.touch_threshold:.3f}м")
        self.logger.info(f"  Мин. площадь: {self.min_touch_area} пикселей")
        self.logger.info(f"  Порог шума: {self.depth_noise_threshold:.3f}м")

    # В методе _preprocess_depth_image замените inpaint на более подходящий метод:

    def _preprocess_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        # Создаем маску валидных значений
        valid_mask = depth_image > 0

        if not np.all(valid_mask):
            # Заполняем нулевые значения медианным фильтром
            kernel_size = 5
            depth_float = depth_image.astype(np.float32)

            # Применяем медианный фильтр только к валидным областям
            if np.any(valid_mask):
                median_filtered = cv2.medianBlur(depth_image, kernel_size)
                depth_image = np.where(valid_mask, depth_image, median_filtered)

        # Гауссово размытие для уменьшения шума
        if self.gaussian_blur_size > 0:
            depth_image = cv2.GaussianBlur(
                depth_image, (self.gaussian_blur_size, self.gaussian_blur_size), 0
            )

        return depth_image

    # def _preprocess_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
    #     """
    #     Предобработка изображения глубины

    #     Args:
    #         depth_image: Исходное изображение глубины

    #     Returns:
    #         Обработанное изображение глубины
    #     """
    #     # Заполняем нулевые значения интерполяцией
    #     mask = depth_image == 0
    #     if np.any(mask):
    #         # Простая интерполяция ближайшими соседями
    #         depth_image = cv2.inpaint(depth_image.astype(np.uint16), mask.astype(np.uint8), 3, cv2.INPAINT_NS)

    #     # Гауссово размытие для уменьшения шума
    #     if self.gaussian_blur_size > 0:
    #         depth_image = cv2.GaussianBlur(depth_image, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

    #     return depth_image

    def _detect_touch_contours(
        self, depth_diff: np.ndarray, current_depth: np.ndarray
    ) -> List[Tuple[int, int, int, float]]:
        """
        Детекция контуров касаний на разностном изображении

        Args:
            depth_diff: Разностное изображение глубины
            current_depth: Текущее изображение глубины для дополнительной фильтрации

        Returns:
            Список кортежей (x, y, area, depth) для каждого касания
        """
        # Используйте адаптивную бинаризацию вместо фиксированного порога
        binary = cv2.adaptiveThreshold(
            (depth_diff * 1000).astype(np.uint8),  # Масштабируем для адаптивного порога
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Размер блока
            2,  # Смещение порога
        )
        # Дополнительная фильтрация по калиброванной поверхности
        if self.surface_height is not None:
            # Конвертируем текущее изображение глубины в метры
            current_depth_meters = current_depth.astype(np.float32) * 0.001

            # Фильтруем только области близкие к поверхности проекции
            surface_mask = (
                np.abs(current_depth_meters - self.surface_height)
                <= self.surface_tolerance
            )
            depth_diff = depth_diff * surface_mask.astype(np.float32)

        # Дополнительная фильтрация шума
        depth_diff[depth_diff < self.depth_noise_threshold] = 0

        # # Пороговая обработка
        # _, binary = cv2.threshold(depth_diff, self.touch_threshold, 255, cv2.THRESH_BINARY)
        # binary = binary.astype(np.uint8)

        # # Морфологическая фильтрация для устранения шума
        # if self.noise_filter_size > 0:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.noise_filter_size, self.noise_filter_size))
        #     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        #     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # # Сохраняем бинарное изображение для отладки
        # self.last_binary_image = binary.copy()
        # Пороговая обработка
        _, binary = cv2.threshold(
            depth_diff, self.touch_threshold, 255, cv2.THRESH_BINARY
        )
        binary = binary.astype(np.uint8)

        # НОВОЕ: Применяем фильтрацию по глубине
        binary = self._apply_depth_filter(current_depth, binary)

        # Морфологическая фильтрация для устранения шума
        if self.noise_filter_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.noise_filter_size, self.noise_filter_size)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Сохраняем бинарное изображение для отладки
        self.last_binary_image = binary.copy()

        # Поиск контуров
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        touches = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Фильтрация по размеру области
            if self.min_touch_area <= area <= self.max_touch_area:
                # Вычисляем центр масс контура
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Применяем пользовательское смещение
                    cx = max(0, min(self.cropped_width - 1, cx + self.touch_offset_x))
                    cy = max(0, min(self.cropped_height - 1, cy + self.touch_offset_y))

                    # Получаем среднюю глубину в области касания
                    mask = np.zeros(depth_diff.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    mean_depth = np.mean(depth_diff[mask > 0])

                    touches.append((cx, cy, int(area), float(mean_depth)))

        return touches

    def set_touch_offset(self, offset_x: int, offset_y: int) -> None:
        self.touch_offset_x = offset_x
        self.touch_offset_y = offset_y
        self.logger.debug(f"Смещение касаний установлено: ({offset_x}, {offset_y})")

    def get_touch_offset_x(self) -> int:
        return self.touch_offset_x

    def get_touch_offset_y(self) -> int:
        return self.touch_offset_y

    def set_depth_filter_range(self, min_depth: float, max_depth: float) -> None:
        """
        Установка диапазона допустимой глубины для касаний

        Args:
            min_depth: Минимальная глубина в метрах
            max_depth: Максимальная глубина в метрах
        """
        self.min_touch_depth = max(0.1, min_depth)  # Не менее 10 см
        self.max_touch_depth = min(10.0, max_depth)  # Не более 10 метров

        self.logger.info(
            f"Диапазон глубины касаний установлен: {self.min_touch_depth:.2f} - {self.max_touch_depth:.2f} м"
        )

    def enable_depth_filter(self, enabled: bool) -> None:
        """
        Включение/выключение фильтрации по глубине

        Args:
            enabled: True для включения фильтрации
        """
        self.enable_depth_filtering = enabled
        self.logger.info(
            f"Фильтрация по глубине: {'включена' if enabled else 'выключена'}"
        )

    def get_depth_filter_stats(self, current_depth: np.ndarray) -> Dict[str, Any]:
        """
        Получение статистики по глубине в текущем кадре

        Args:
            current_depth: Текущее изображение глубины

        Returns:
            Словарь со статистикой глубины
        """
        depth_meters = current_depth.astype(np.float32) * 0.001
        valid_pixels = depth_meters[
            (depth_meters > 0) & (depth_meters < 10)
        ]  # Исключаем 0 и большие значения

        if len(valid_pixels) == 0:
            return {"valid_pixels": 0}

        return {
            "valid_pixels": len(valid_pixels),
            "min_depth": float(np.min(valid_pixels)),
            "max_depth": float(np.max(valid_pixels)),
            "mean_depth": float(np.mean(valid_pixels)),
            "median_depth": float(np.median(valid_pixels)),
            "depth_range_set": (self.min_touch_depth, self.max_touch_depth),
            "filter_enabled": self.enable_depth_filtering,
        }

    def _scale_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """
        Масштабирование координат с кропнутого изображения до целевого разрешения

        Args:
            x: X координата в кропнутом изображении
            y: Y координата в кропнутом изображении

        Returns:
            Кортеж (scaled_x, scaled_y) в целевом разрешении
        """
        scaled_x = int(x * self.scale_x)
        scaled_y = int(y * self.scale_y)

        # Ограничиваем координаты в пределах целевого разрешения
        scaled_x = max(0, min(scaled_x, self.target_width - 1))
        scaled_y = max(0, min(scaled_y, self.target_height - 1))

        return scaled_x, scaled_y

    def _stabilize_touches(self, current_touches: List[TouchPoint]) -> List[TouchPoint]:
        """
        Стабилизация касаний с использованием истории

        Args:
            current_touches: Текущие обнаруженные касания

        Returns:
            Стабилизированный список касаний
        """
        # Добавляем текущие касания в историю
        self.touch_history.append(current_touches)

        # Ограничиваем размер истории
        if len(self.touch_history) > self.history_length:
            self.touch_history.pop(0)

        # Если недостаточно истории, возвращаем текущие касания
        if len(self.touch_history) < 2:
            return current_touches

        # Простая стабилизация: касание должно присутствовать в большинстве кадров
        stable_touches = []
        for touch in current_touches:
            # Проверяем, есть ли похожие касания в предыдущих кадрах
            similar_count = 0
            for hist_touches in self.touch_history[:-1]:  # Исключаем текущий кадр
                for hist_touch in hist_touches:
                    # Вычисляем расстояние между касаниями
                    distance = np.sqrt(
                        (touch.x - hist_touch.x) ** 2 + (touch.y - hist_touch.y) ** 2
                    )
                    if distance < 30:  # Порог похожести в пикселях
                        similar_count += 1
                        break

            # Если касание стабильно, добавляем его
            if similar_count >= len(self.touch_history) // 2:
                stable_touches.append(touch)

        return stable_touches

    def _filter_touches(self, touches: List[TouchPoint]) -> List[TouchPoint]:
        """
        Фильтрация касаний с использованием продвинутого алгоритма

        Args:
            touches: Список обнаруженных касаний

        Returns:
            Отфильтрованный список касаний
        """
        if not self.use_advanced_filter or not touches:
            return touches

        # Конвертируем TouchPoint в формат для TouchFilter
        touch_data = []
        for touch in touches:
            touch_data.append(
                (
                    touch.x,
                    touch.y,
                    {
                        "depth": touch.depth,
                        "area": touch.area,
                        "confidence": touch.confidence,
                        "timestamp": touch.timestamp,
                    },
                )
            )

        # Применяем продвинутую фильтрацию
        filtered_data = self.touch_filter.update(touch_data)

        # Конвертируем обратно в TouchPoint
        filtered_touches = []
        for x, y, info in filtered_data:
            filtered_touch = TouchPoint(
                x=int(x),
                y=int(y),
                depth=info["depth"],
                area=info["area"],
                confidence=info["confidence"],
            )
            filtered_touches.append(filtered_touch)

        return filtered_touches

    def process_frame(
        self,
        cropped_color: np.ndarray,
        cropped_depth: np.ndarray,
        auto_update_background: bool = True,
    ) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Обработка кадра для детекции касаний

        Args:
            cropped_color: Кропнутое цветное изображение
            cropped_depth: Кропнутое изображение глубины
            auto_update_background: Автоматически обновлять фон

        Returns:
            Список кортежей (scaled_x, scaled_y, touch_info) для каждого касания
        """
        self.frame_count += 1

        # Проверяем размеры изображений
        if cropped_depth.shape != (self.cropped_height, self.cropped_width):
            self.logger.error(
                f"Размер изображения глубины {cropped_depth.shape} не соответствует ожидаемому"
            )
            return []

        # Предобработка изображения глубины
        processed_depth = self._preprocess_depth_image(cropped_depth)

        # Применяем коррекцию глубины для устранения параллакса камера-проектор
        processed_depth = self.virtual_camera_distance_adjustment(processed_depth)

        # Инициализация или обновление фона
        if auto_update_background:
            self.update_background(processed_depth)

        if self.background_depth is None:
            return []

        # # Вычисляем разность с фоном
        # depth_diff = self.background_depth.astype(np.float32) + processed_depth.astype(np.float32) - self.background_threshold
        # # depth_diff = processed_depth.astype(np.float32) - self.background_depth.astype(np.float32)

        # # Нормализуем разность (глубина в миллиметрах)
        # depth_diff = depth_diff * 0.001  # Преобразуем в метры
        # ВЫЧИСЛЕНИЕ РАЗНОСТИ С УЧЕТОМ ПЛОСКОСТИ
        if self.plane_A is not None:
            # Используем разность с плоскостью поверхности
            current_depth_meters = processed_depth.astype(np.float32) * 0.001

            # Вычисляем ожидаемую поверхность для каждого пикселя
            height, width = current_depth_meters.shape
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            expected_surface = (
                self.plane_A * x_coords + self.plane_B * y_coords + self.plane_C
            )

            # Разность: положительная = объект ближе к камере (касание)
            depth_diff = expected_surface - current_depth_meters
            depth_diff[depth_diff < 0] = 0  # Игнорируем объекты дальше поверхности

        else:
            # Fallback: старый метод (разность с фоном)
            depth_diff = self.background_depth.astype(
                np.float32
            ) - processed_depth.astype(np.float32)
            depth_diff = depth_diff * 0.001  # в метры

        # Сохраняем разностное изображение для отладки
        self.last_depth_diff = depth_diff.copy()

        # Детекция касаний
        touch_contours = self._detect_touch_contours(depth_diff, processed_depth)

        # Создаем объекты TouchPoint
        current_touches = []
        for x, y, area, mean_depth in touch_contours:
            confidence = min(
                1.0, area / self.min_touch_area
            )  # Простая оценка уверенности
            touch = TouchPoint(x, y, mean_depth, area, confidence)
            current_touches.append(touch)

        # Стабилизация касаний
        stable_touches = self._stabilize_touches(current_touches)

        # Масштабирование координат и подготовка результата
        result = []
        for touch in stable_touches:
            scaled_x, scaled_y = self._scale_coordinates(touch.x, touch.y)

            touch_info = {
                "original_x": touch.x,
                "original_y": touch.y,
                "depth": touch.depth,
                "area": touch.area,
                "confidence": touch.confidence,
                "timestamp": touch.timestamp,
            }

            result.append((scaled_x, scaled_y, touch_info))

        self.total_touches_detected += len(result)

        if result:
            self.logger.debug(f"Обнаружено касаний: {len(result)}")

        return result

    def get_expected_depth(self, x: int, y: int) -> Optional[float]:
        """Получить ожидаемую глубину поверхности в точке (x, y)"""
        if self.plane_A is None:
            return None
        return self.plane_A * x + self.plane_B * y + self.plane_C

    def get_depth_deviation(
        self, x: int, y: int, actual_depth: float
    ) -> Optional[float]:
        """Получить отклонение от ожидаемой поверхности"""
        expected = self.get_expected_depth(x, y)
        if expected is None:
            return None
        return actual_depth - expected  # Отрицательное = ближе к камере

    def get_plane_calibration_quality(self) -> Dict[str, Any]:
        """Оценка качества калибровки плоскости"""
        if self.plane_A is None:
            return {"calibrated": False}

        quality = {
            "calibrated": True,
            "rmse": self.plane_rmse,
            "max_error": self.plane_max_error,
            "points_count": len(self.surface_calibration_points),
            "equation": f"z = {self.plane_A:.4f}x + {self.plane_B:.4f}y + {self.plane_C:.3f}",
        }

        # Оценка качества
        if self.plane_rmse < 0.01:
            quality["quality"] = "excellent"
        elif self.plane_rmse < 0.02:
            quality["quality"] = "good"
        elif self.plane_rmse < 0.05:
            quality["quality"] = "fair"
        else:
            quality["quality"] = "poor"

        return quality

    def visualize_touches(
        self, cropped_color: np.ndarray, touches: List[Tuple[int, int, Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Визуализация касаний на кропнутом изображении

        Args:
            cropped_color: Кропнутое цветное изображение
            touches: Список касаний с информацией

        Returns:
            Изображение с визуализированными касаниями
        """
        result = cropped_color.copy()

        for i, (scaled_x, scaled_y, touch_info) in enumerate(touches):
            # Получаем исходные координаты в кропнутом изображении
            orig_x = touch_info["original_x"]
            orig_y = touch_info["original_y"]
            confidence = touch_info["confidence"]
            area = touch_info["area"]

            # Цвет зависит от уверенности
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))

            # Рисуем круг в точке касания
            cv2.circle(result, (orig_x, orig_y), 10, color, -1)
            cv2.circle(result, (orig_x, orig_y), 15, color, 2)

            # Добавляем номер касания
            cv2.putText(
                result,
                str(i + 1),
                (orig_x + 20, orig_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # Добавляем информацию о касании
            info_text = f"({scaled_x},{scaled_y}) A:{area}"
            cv2.putText(
                result,
                info_text,
                (orig_x + 20, orig_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Добавляем общую информацию
        cv2.putText(
            result,
            f"Touches: {len(touches)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            result,
            f"Total: {self.total_touches_detected}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        return result

    def reset_background(self) -> None:
        """
        Сброс фонового изображения
        """
        self.background_depth = None
        self.touch_history.clear()
        self.logger.info("Фоновое изображение сброшено")

    def reset_all(self) -> None:
        """
        Полный сброс всех настроек и калибровки
        """
        self.reset_background()
        self.clear_surface_calibration()
        self.sensitivity_level = 5
        self.adjust_sensitivity(0)  # Пересчитать параметры для уровня 5
        self.logger.info("Все настройки сброшены к значениям по умолчанию")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики работы процессора

        Returns:
            Словарь со статистической информацией
        """
        stats = {
            "frame_count": self.frame_count,
            "total_touches": self.total_touches_detected,
            "average_touches_per_frame": self.total_touches_detected
            / max(1, self.frame_count),
            "has_background": self.background_depth is not None,
            "cropped_resolution": (self.cropped_width, self.cropped_height),
            "target_resolution": (self.target_width, self.target_height),
            "scale_factors": (self.scale_x, self.scale_y),
            "touch_threshold": self.touch_threshold,
            "min_touch_area": self.min_touch_area,
            "max_touch_area": self.max_touch_area,
            "sensitivity_level": self.sensitivity_level,
            "surface_calibrated": self.surface_height is not None,
            "surface_height": self.surface_height,
            "surface_calibration_points": len(self.surface_calibration_points),
            "depth_noise_threshold": self.depth_noise_threshold,
            "surface_plane_calibrated": self.plane_A is not None,
            "plane_rmse": self.plane_rmse if self.plane_A else None,
            "plane_max_error": self.plane_max_error if self.plane_A else None,
            "depth_filter_enabled": self.enable_depth_filtering,
            "min_touch_depth": self.min_touch_depth,
            "max_touch_depth": self.max_touch_depth,
        }
        if self.plane_A:
            stats.update(self.get_plane_calibration_quality())
        return stats

    def configure_detection(
        self,
        touch_threshold: Optional[float] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        background_update_rate: Optional[float] = None,
    ) -> None:
        """
        Настройка параметров детекции

        Args:
            touch_threshold: Порог детекции касания в метрах
            min_area: Минимальная площадь касания в пикселях
            max_area: Максимальная площадь касания в пикселях
            background_update_rate: Скорость обновления фона (0.0-1.0)
        """
        if touch_threshold is not None:
            self.touch_threshold = touch_threshold
            self.logger.info(f"Порог касания установлен: {touch_threshold:.3f}м")

        if min_area is not None:
            self.min_touch_area = min_area
            self.logger.info(f"Минимальная площадь касания: {min_area} пикселей")

        if max_area is not None:
            self.max_touch_area = max_area
            self.logger.info(f"Максимальная площадь касания: {max_area} пикселей")

        if background_update_rate is not None:
            self.background_update_rate = max(0.0, min(1.0, background_update_rate))
            self.logger.info(
                f"Скорость обновления фона: {self.background_update_rate:.3f}"
            )

    def get_debug_images(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Получение отладочных изображений для визуализации процесса детекции

        Returns:
            Словарь с отладочными изображениями:
            - 'binary': бинарное изображение зоны детекции
            - 'depth_diff': разностное изображение глубины
            - 'depth_diff_normalized': нормализованное разностное изображение для отображения
        """
        result = {
            "binary": self.last_binary_image,
            "depth_diff": self.last_depth_diff,
            "depth_diff_normalized": None,
        }

        # Создаем нормализованное изображение разности для визуализации
        if self.last_depth_diff is not None:
            # Нормализуем для отображения (0-255)
            diff_normalized = self.last_depth_diff.copy()

            # Ограничиваем диапазон для лучшей визуализации
            diff_normalized = np.clip(diff_normalized, 0, self.touch_threshold * 3)

            # Преобразуем в диапазон 0-255
            if np.max(diff_normalized) > 0:
                diff_normalized = (
                    diff_normalized / np.max(diff_normalized) * 255
                ).astype(np.uint8)
            else:
                diff_normalized = np.zeros_like(diff_normalized, dtype=np.uint8)

            result["depth_diff_normalized"] = diff_normalized

        return result

    def adjust_depth_perception(
        self,
        depth_image: np.ndarray,
        distance_offset_mm: float = None,
        scale_factor: float = None,
    ) -> np.ndarray:
        """
        Виртуально "отдаляет" камеру путем добавления смещения к значениям глубины

        Args:
            depth_image: Исходное изображение глубины
            distance_offset_mm: На сколько мм виртуально отдаляем камеру (если None, использует self.depth_offset_mm)
            scale_factor: Коэффициент масштабирования (если None, использует self.depth_scale_factor)

        Returns:
            Скорректированное изображение глубины
        """
        if distance_offset_mm is None:
            distance_offset_mm = self.depth_offset_mm
        if scale_factor is None:
            scale_factor = self.depth_scale_factor

        # Добавляем смещение ко всем значениям глубины
        adjusted_depth = depth_image.astype(np.float32) + distance_offset_mm

        # Ограничиваем значения максимальным диапазоном камеры
        adjusted_depth = np.clip(adjusted_depth, 0, self.max_depth_mm)

        return adjusted_depth.astype(np.uint16)

    def scale_depth_perception(
        self, depth_image: np.ndarray, scale_factor: float = None
    ) -> np.ndarray:
        """
        Виртуально "отдаляет" камеру через масштабирование значений глубины

        Args:
            depth_image: Исходное изображение глубины
            scale_factor: Коэффициент масштабирования (>1 отдаляет камеру)

        Returns:
            Масштабированное изображение глубины
        """
        if scale_factor is None:
            scale_factor = self.depth_scale_factor

        scaled_depth = depth_image.astype(np.float32) * scale_factor
        return np.clip(scaled_depth, 0, self.max_depth_mm).astype(np.uint16)

    def spatial_depth_filter(
        self, depth_image: np.ndarray, kernel_size: int = None
    ) -> np.ndarray:
        """
        Применяет медианный фильтр для уменьшения шума в данных глубины

        Args:
            depth_image: Исходное изображение глубины
            kernel_size: Размер ядра фильтра (если None, использует self.spatial_filter_kernel)

        Returns:
            Отфильтрованное изображение глубины
        """
        if kernel_size is None:
            kernel_size = self.spatial_filter_kernel

        # Убеждаемся, что размер ядра нечетный и в допустимом диапазоне
        kernel_size = max(3, min(15, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.medianBlur(depth_image, kernel_size)

    def virtual_camera_distance_adjustment(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Комплексная коррекция для виртуального отдаления камеры

        Args:
            depth_image: Исходное изображение глубины

        Returns:
            Скорректированное изображение глубины
        """
        # Применяем фильтрацию если включена
        if self.enable_spatial_filter:
            depth_image = self.spatial_depth_filter(depth_image)

        # Масштабируем глубину
        scaled_depth = depth_image.astype(np.float32) * self.depth_scale_factor

        # Добавляем смещение
        adjusted_depth = scaled_depth + self.depth_offset_mm

        # Ограничиваем диапазон
        return np.clip(adjusted_depth, 0, self.max_depth_mm).astype(np.uint16)

    def set_background_threshold(self, threshold: float) -> None:
        """
        Устанавливает новое значение порога фона

        Args:
            threshold: Новое значение порога в метрах (-1.0 до 3.0)
        """
        # Ограничиваем значение допустимым диапазоном
        threshold = max(-1.0, min(3.0, threshold))
        self.background_threshold = threshold
        self.logger.debug(f"Порог фона изменен на: {threshold:.3f}м")

    def get_background_threshold(self) -> float:
        """
        Возвращает текущее значение порога фона

        Returns:
            Текущий порог фона в метрах
        """
        return self.background_threshold

    def set_touch_threshold(self, threshold: float) -> None:
        """
        Устанавливает новое значение порога касания

        Args:
            threshold: Новое значение порога в метрах (0.0 до 0.5)
        """
        # Ограничиваем значение допустимым диапазоном
        threshold = max(0.0, min(0.5, threshold))
        self.touch_threshold = threshold
        self.logger.debug(f"Порог касания изменен на: {threshold:.3f}м")

    def get_touch_threshold(self) -> float:
        """
        Возвращает текущее значение порога касания

        Returns:
            Текущий порог касания в метрах
        """
        return self.touch_threshold

    # Методы управления коррекцией глубины
    def set_depth_offset(self, offset_mm: float) -> None:
        """
        Устанавливает смещение глубины для коррекции параллакса

        Args:
            offset_mm: Смещение в миллиметрах (-100 до +200)
        """
        self.depth_offset_mm = max(-100.0, min(200.0, offset_mm))
        self.logger.debug(f"Смещение глубины изменено на: {self.depth_offset_mm:.1f}мм")

    def get_depth_offset(self) -> float:
        """Возвращает текущее смещение глубины в мм"""
        return self.depth_offset_mm

    def set_depth_scale_factor(self, scale_factor: float) -> None:
        """
        Устанавливает коэффициент масштабирования глубины

        Args:
            scale_factor: Коэффициент масштабирования (0.8 до 1.5)
        """
        self.depth_scale_factor = max(0.8, min(1.5, scale_factor))
        self.logger.debug(
            f"Коэффициент масштабирования изменен на: {self.depth_scale_factor:.3f}"
        )

    def get_depth_scale_factor(self) -> float:
        """Возвращает текущий коэффициент масштабирования глубины"""
        return self.depth_scale_factor

    def set_spatial_filter_enabled(self, enabled: bool) -> None:
        """
        Включает/выключает пространственную фильтрацию

        Args:
            enabled: True для включения фильтрации
        """
        self.enable_spatial_filter = enabled
        self.logger.debug(
            f"Пространственная фильтрация: {'включена' if enabled else 'выключена'}"
        )

    def get_spatial_filter_enabled(self) -> bool:
        """Возвращает состояние пространственной фильтрации"""
        return self.enable_spatial_filter

    def set_spatial_filter_kernel(self, kernel_size: int) -> None:
        """
        Устанавливает размер ядра пространственной фильтрации

        Args:
            kernel_size: Размер ядра (3, 5, 7, 9, 11, 13, 15)
        """
        # Ограничиваем диапазон и делаем размер нечетным
        kernel_size = max(3, min(15, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.spatial_filter_kernel = kernel_size
        self.logger.debug(f"Размер ядра фильтра изменен на: {kernel_size}")

    def get_spatial_filter_kernel(self) -> int:
        """Возвращает размер ядра пространственной фильтрации"""
        return self.spatial_filter_kernel

    # Методы управления TouchFilter
    def set_advanced_filter_enabled(self, enabled: bool) -> None:
        """
        Включает/выключает продвинутую фильтрацию касаний

        Args:
            enabled: True для включения продвинутой фильтрации
        """
        self.use_advanced_filter = enabled
        self.logger.debug(
            f"Продвинутая фильтрация касаний: {'включена' if enabled else 'выключена'}"
        )

    def get_advanced_filter_enabled(self) -> bool:
        """Возвращает состояние продвинутой фильтрации"""
        return self.use_advanced_filter

    def set_filter_distance_threshold(self, threshold: float) -> None:
        """
        Устанавливает пороговое расстояние для фильтрации

        Args:
            threshold: Пороговое расстояние в пикселях (5-100)
        """
        self.touch_filter.set_threshold(threshold)
        self.logger.debug(f"Порог фильтрации касаний изменен на: {threshold}px")

    def get_filter_distance_threshold(self) -> float:
        """Возвращает текущий порог фильтрации"""
        return self.touch_filter.distance_threshold

    def set_filter_min_movement(self, min_movement: float) -> None:
        """
        Устанавливает минимальное движение для регистрации

        Args:
            min_movement: Минимальное движение в пикселях (1-50)
        """
        self.touch_filter.set_min_movement(min_movement)
        self.logger.debug(f"Минимальное движение изменено на: {min_movement}px")

    def get_filter_min_movement(self) -> float:
        """Возвращает минимальное движение"""
        return self.touch_filter.min_movement_threshold

    def set_filter_history_size(self, history_size: int) -> None:
        """
        Устанавливает размер истории фильтра

        Args:
            history_size: Размер истории (2-20)
        """
        self.touch_filter.set_history_size(history_size)
        self.logger.debug(f"Размер истории фильтра изменен на: {history_size}")

    def get_filter_history_size(self) -> int:
        """Возвращает размер истории фильтра"""
        return self.touch_filter.history_size

    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики фильтра касаний

        Returns:
            Словарь со статистикой фильтра
        """
        return self.touch_filter.get_statistics()

    def reset_filter(self) -> None:
        """Сброс состояния фильтра касаний"""
        self.touch_filter.reset()
        self.logger.info("Состояние фильтра касаний сброшено")

    def set_filter_timeout(self, timeout: float) -> None:
        """
        Устанавливает таймаут для статичных касаний

        Args:
            timeout: Время в секундах, после которого статичное касание скрывается (0.1-10.0)
        """
        self.touch_filter.set_timeout(timeout)
        self.logger.debug(f"Таймаут касаний изменен на: {timeout}с")

    def get_filter_timeout(self) -> float:
        """Возвращает текущий таймаут касаний"""
        return self.touch_filter.get_timeout()

    def get_touch_durations(self) -> Dict[int, float]:
        """
        Получение длительности активных касаний

        Returns:
            Словарь {touch_id: duration_in_seconds}
        """
        return self.touch_filter.get_touch_durations()


def test_touch_processor() -> None:
    """
    Тест процессора касаний
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    # Создаем тестовый процессор
    processor = TouchProcessor(800, 600, 1920, 1080)

    # Создаем тестовые изображения
    test_color = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    test_depth = np.random.randint(500, 1500, (600, 800), dtype=np.uint16)

    # Устанавливаем фон
    processor.set_background(test_depth)

    # Имитируем касание
    test_depth_with_touch = test_depth.copy()
    test_depth_with_touch[200:250, 300:350] = 400  # Имитация касания

    # Обрабатываем кадр
    touches = processor.process_frame(test_color, test_depth_with_touch)

    print(f"Обнаружено касаний: {len(touches)}")
    for i, (x, y, info) in enumerate(touches):
        print(f"Касание {i+1}: ({x}, {y}) - {info}")

    # Показываем статистику
    stats = processor.get_statistics()
    print(f"Статистика: {stats}")


if __name__ == "__main__":
    test_touch_processor()
