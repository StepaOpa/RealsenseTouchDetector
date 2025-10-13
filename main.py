"""
Основной скрипт для системы распознавания касаний с камерой Intel RealSense D435F
Получает и отображает цветные изображения и изображения глубины
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import sys
import logging
from realsense_camera import RealSenseCamera
from calibrate_projection_area import ProjectionCalibrator
from crop_to_calibrated_area import ProjectionAreaCropper
from touch_processor import TouchProcessor
from unity_communication import UnityCommunication


class TouchDetector:
    """
    Класс для детекции касаний на проекции
    """
    
    def __init__(self, camera: RealSenseCamera) -> None:
        """
        Инициализация детектора касаний
        
        Args:
            camera: Экземпляр камеры RealSense
        """
        self.camera: RealSenseCamera = camera
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.window_name: str = 'RealSense D435F - Touch Detection'
        
        self.depth_filter_enabled: bool = True  # Включена ли фильтрация по глубине
        self.min_touch_depth: float = 0.3  # Минимальная глубина касания в метрах (30 см)
        self.max_touch_depth: float = 2.0  # Максимальная глубина касания в метрах (2 метра)
        
        # Калибратор проекции
        self.calibrator: Optional[ProjectionCalibrator] = None
        self.is_calibration_mode: bool = False
        
        # Обрезчик проекции
        self.cropper: ProjectionAreaCropper = ProjectionAreaCropper(output_size=(800,600))
        self.show_cropped: bool = False  # Режим отображения: False - исходные, True - обрезанные
        self.show_projection_area: bool = True  # Показывать ли область проекции на исходном изображении
        
        # Процессор касаний
        self.touch_processor: Optional[TouchProcessor] = None
        self.touch_detection_enabled: bool = False  # Включена ли детекция касаний
        self.show_touches: bool = True  # Показывать ли касания на изображении
        self.current_touches: List = []  # Текущие обнаруженные касания
        self.show_debug_images: bool = False  # Показывать ли отладочные изображения детекции
        
        # Unity коммуникация
        self.unity_comm: Optional[UnityCommunication] = None
        self.unity_enabled: bool = True  # Включена ли отправка в Unity
        self.unity_host: str = "127.0.0.1"  # IP адрес Unity
        self.unity_port: int = 8052  # UDP порт для Unity
        
        # Калибровка поверхности
        self.surface_calibration_mode: bool = False  # Режим калибровки поверхности
        self.surface_calibration_points_needed: int = 5  # Количество точек для калибровки
        
        # Инициализация камеры
        self._setup_logging()
        
        # Инициализируем процессор касаний если есть калибровка
        self._init_touch_processor()
        
        # Настройка обработчика мыши
        self._setup_mouse_callback()
        
        # Создание окна управления с ползунками
        self._setup_control_window()
        
        self.show_depth = True
    
    def _setup_logging(self) -> None:
        """
        Настройка логирования
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _init_touch_processor(self) -> None:
        """
        Инициализация процессора касаний
        """
        if self.cropper.is_calibrated:
            self.touch_processor = TouchProcessor(
                cropped_width=800,
                cropped_height=600,
                target_width=1920,
                target_height=1080
            )
            self.logger.info("TouchProcessor инициализирован")
            
            # НАСТРАИВАЕМ ФИЛЬТРАЦИЮ ПО ГЛУБИНЕ
            self.touch_processor.enable_depth_filter(self.depth_filter_enabled)
            self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
            self.logger.info(f"Фильтрация по глубине: {'включена' if self.depth_filter_enabled else 'выключена'}")
            self.logger.info(f"Диапазон глубины: {self.min_touch_depth:.2f}-{self.max_touch_depth:.2f}м")
            
            # Инициализируем Unity коммуникацию
            self._init_unity_communication()
        else:
            self.logger.info("TouchProcessor не инициализирован - нет калибровки")
    
    def _init_unity_communication(self) -> None:
        """
        Инициализация Unity коммуникации
        """
        try:
            self.unity_comm = UnityCommunication(
                host=self.unity_host,
                port=self.unity_port,
                target_width=1920,
                target_height=1080
            )
            
            # Тестируем соединение
            if self.unity_comm.test_connection():
                self.logger.info(f"Unity коммуникация инициализирована: {self.unity_host}:{self.unity_port}")
            else:
                self.logger.warning("Unity приложение не отвечает, но модуль готов к работе")
                
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Unity коммуникации: {e}")
            self.unity_comm = None
    
    def _setup_mouse_callback(self) -> None:
        """
        Настройка обработчика мыши для калибровки поверхности
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def _setup_control_window(self) -> None:
        """
        Создание окна управления с ползунками для настройки параметров детекции
        """
        self.control_window_name = "Управление детекцией"
        cv2.namedWindow(self.control_window_name, cv2.WINDOW_AUTOSIZE)
        
        # Создаем информационное изображение для окна управления
        control_image = np.zeros((550, 500, 3), dtype=np.uint8)
        cv2.putText(control_image, "Touch Detection Settings", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.touch_processor is not None:
            # Ползунок для background_threshold (-1 до 3 метров)
            # OpenCV ползунки работают только с целыми числами, поэтому используем миллиметры
            current_bg_threshold = int(self.touch_processor.get_background_threshold() * 1000)  # Конвертируем в мм
            cv2.createTrackbar(
                "background threshold (mm)",
                self.control_window_name,
                current_bg_threshold + 1000,  # Смещаем на 1000, чтобы -1000мм стало 0
                4000,  # Диапазон: от 0 (=-1000мм) до 4000 (=3000мм)
                self._on_background_threshold_change
            )
            
            # Ползунок для touch_threshold (0 до 0.5 метров = 0 до 500 мм)
            current_touch_threshold = int(self.touch_processor.get_touch_threshold() * 1000)
            cv2.createTrackbar(
                "touch threshold (mm)",
                self.control_window_name,
                current_touch_threshold,
                500,  # 0 до 500мм
                self._on_touch_threshold_change
            )
            
            # Ползунок для смещения глубины (-100 до +200 мм)
            current_depth_offset = int(self.touch_processor.get_depth_offset())
            cv2.createTrackbar(
                "depth offset (mm)",
                self.control_window_name,
                current_depth_offset + 100,  # Смещаем на 100, чтобы -100мм стало 0
                300,  # Диапазон: от 0 (=-100мм) до 300 (=200мм)
                self._on_depth_offset_change
            )
            
            # Ползунок для масштабирования глубины (0.8 до 1.5)
            current_scale_factor = int(self.touch_processor.get_depth_scale_factor() * 100)
            cv2.createTrackbar(
                "depth scale (%)",
                self.control_window_name,
                current_scale_factor,  # 80 до 150 (представляет 0.8 до 1.5)
                150,
                self._on_depth_scale_change
            )
            
            # Ползунок для размера ядра пространственной фильтрации (3 до 15)
            current_filter_kernel = self.touch_processor.get_spatial_filter_kernel()
            cv2.createTrackbar(
                "filter kernel",
                self.control_window_name,
                current_filter_kernel,
                15,  # 3 до 15
                self._on_spatial_filter_kernel_change
            )
            
            # Ползунок для включения/выключения пространственной фильтрации
            filter_enabled = 1 if self.touch_processor.get_spatial_filter_enabled() else 0
            cv2.createTrackbar(
                "filter on/off",
                self.control_window_name,
                filter_enabled,
                1,  # 0 или 1
                self._on_spatial_filter_toggle
            )
            
            # Ползунок для включения/выключения Unity передачи
            unity_enabled = 1 if self.unity_enabled else 0
            cv2.createTrackbar(
                "Unity send",
                self.control_window_name,
                unity_enabled,
                1,  # 0 или 1
                self._on_unity_enabled_toggle
            )
            
            # Ползунок для Unity порта (8000-9000)
            cv2.createTrackbar(
                "Unity port",
                self.control_window_name,
                self.unity_port - 8000,  # Смещаем базу на 8000
                1000,  # 8000 до 9000
                self._on_unity_port_change
            )
            
            # Ползунки для TouchFilter
            # Включение/выключение продвинутой фильтрации
            advanced_filter_enabled = 1 if self.touch_processor.get_advanced_filter_enabled() else 0
            cv2.createTrackbar(
                "advanced filter",
                self.control_window_name,
                advanced_filter_enabled,
                1,  # 0 или 1
                self._on_advanced_filter_toggle
            )
            
            # Пороговое расстояние для фильтрации (5-100 пикселей)
            current_filter_threshold = int(self.touch_processor.get_filter_distance_threshold())
            cv2.createTrackbar(
                "filter threshold",
                self.control_window_name,
                current_filter_threshold - 5,  # Смещаем базу на 5
                95,  # 5 до 100
                self._on_filter_threshold_change
            )
            
            # Минимальное движение (1-50 пикселей)
            current_min_movement = int(self.touch_processor.get_filter_min_movement())
            cv2.createTrackbar(
                "min movement",
                self.control_window_name,
                current_min_movement - 1,  # Смещаем базу на 1
                49,  # 1 до 50
                self._on_min_movement_change
            )
            
            # Таймаут для статичных касаний (0.1-10.0 секунд)
            current_timeout = int(self.touch_processor.get_filter_timeout() * 10)  # Конвертируем в десятые доли
            cv2.createTrackbar(
                "touch timeout",
                self.control_window_name,
                current_timeout - 1,  # Смещаем базу на 1 (0.1с)
                99,  # 0.1 до 10.0 секунд
                self._on_timeout_change
            )
                        
            # Ползунок для включения/выключения фильтрации по глубине
            depth_filter_enabled = 1 if self.depth_filter_enabled else 0
            cv2.createTrackbar(
                "depth filter on/off",
                self.control_window_name,
                depth_filter_enabled,
                1,  # 0 или 1
                self._on_depth_filter_toggle
            )
            
            # Ползунок для минимальной глубины (10см до 2м = 100мм до 2000мм)
            current_min_depth = int(self.min_touch_depth * 1000)  # Конвертируем в мм
            cv2.createTrackbar(
                "min depth (mm)",
                self.control_window_name,
                current_min_depth - 100,  # Смещаем базу на 100 (100мм->0)
                4000,  # 100мм до 2000мм
                self._on_min_depth_change
            )
            
            # Ползунок для максимальной глубины (0.5м до 5м = 500мм до 5000мм)
            current_max_depth = int(self.max_touch_depth * 1000)  # Конвертируем в мм
            cv2.createTrackbar(
                "max depth (mm)",
                self.control_window_name,
                current_max_depth - 500,  # Смещаем базу на 500 (500мм->0)
                4500,  # 500мм до 5000мм
                self._on_max_depth_change
            )
                
            # Добавляем информацию о текущих значениях
            y_pos = 70
            line_height = 25
            
            # Основные пороги
            cv2.putText(control_image, f"Background Threshold: {self.touch_processor.get_background_threshold():.3f}m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_pos += line_height
            
            cv2.putText(control_image, f"Touch Threshold: {self.touch_processor.get_touch_threshold():.3f}m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_pos += line_height
            
            # Коррекция глубины
            cv2.putText(control_image, f"Depth Offset: {self.touch_processor.get_depth_offset():.1f}mm", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            y_pos += line_height
            
            cv2.putText(control_image, f"Depth Scale: {self.touch_processor.get_depth_scale_factor():.2f}x", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            y_pos += line_height
            
            # Фильтрация
            filter_status = "ON" if self.touch_processor.get_spatial_filter_enabled() else "OFF"
            cv2.putText(control_image, f"Spatial Filter: {filter_status} (kernel: {self.touch_processor.get_spatial_filter_kernel()})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_pos += line_height
            
            # Разделительная линия
            cv2.line(control_image, (10, y_pos + 5), (440, y_pos + 5), (100, 100, 100), 1)
            y_pos += 20
            
            # Диапазоны значений
            cv2.putText(control_image, "Ranges:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += line_height
            
            cv2.putText(control_image, "Background: -1.0m ... +3.0m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Touch: 0.0m ... 0.5m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Depth Offset: -100mm ... +200mm", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Depth Scale: 0.8x ... 1.5x", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            self.logger.info("Окно управления создано")
        else:
            cv2.putText(control_image, "TouchProcessor не инициализирован", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(control_image, "Сначала выполните калибровку", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self.logger.warning("TouchProcessor не инициализирован - окно управления не создано")
        
        # Отображаем информационное изображение
        cv2.imshow(self.control_window_name, control_image)
    
    
    def _on_background_threshold_change(self, value: int) -> None:
        """
        Callback для изменения порога фона
        
        Args:
            value: Значение от 0 до 4000 (представляет от -1000мм до +3000мм)
        """
        if self.touch_processor is not None:
            # Конвертируем обратно в метры: 0->-1.0м, 1000->0.0м, 4000->3.0м
            threshold_meters = (value - 1000) / 1000.0
            self.touch_processor.set_background_threshold(threshold_meters)
            self.logger.debug(f"Порог фона изменен: {threshold_meters:.3f}м")
            self._update_control_window()
    
    def _on_touch_threshold_change(self, value: int) -> None:
        """
        Callback для изменения порога касания
        
        Args:
            value: Значение от 0 до 500 (представляет от 0мм до 500мм)
        """
        if self.touch_processor is not None:
            # Конвертируем в метры
            threshold_meters = value / 1000.0
            self.touch_processor.set_touch_threshold(threshold_meters)
            self.logger.debug(f"Порог касания изменен: {threshold_meters:.3f}м")
            self._update_control_window()
    
    def _on_depth_offset_change(self, value: int) -> None:
        """
        Callback для изменения смещения глубины
        
        Args:
            value: Значение от 0 до 300 (представляет от -100мм до +200мм)
        """
        if self.touch_processor is not None:
            # Конвертируем обратно в мм: 0->-100мм, 100->0мм, 300->200мм
            offset_mm = value - 100
            self.touch_processor.set_depth_offset(offset_mm)
            self.logger.debug(f"Смещение глубины изменено: {offset_mm:.1f}мм")
            self._update_control_window()
    
    def _on_depth_scale_change(self, value: int) -> None:
        """
        Callback для изменения масштабирования глубины
        
        Args:
            value: Значение от 80 до 150 (представляет от 0.8 до 1.5)
        """
        if self.touch_processor is not None:
            # Конвертируем в коэффициент: 80->0.8, 100->1.0, 150->1.5
            scale_factor = max(80, min(150, value)) / 100.0
            self.touch_processor.set_depth_scale_factor(scale_factor)
            self.logger.debug(f"Коэффициент масштабирования изменен: {scale_factor:.2f}")
            self._update_control_window()
    
    def _on_spatial_filter_kernel_change(self, value: int) -> None:
        """
        Callback для изменения размера ядра пространственной фильтрации
        
        Args:
            value: Размер ядра от 3 до 15
        """
        if self.touch_processor is not None:
            kernel_size = max(3, min(15, value))
            # Убеждаемся, что размер нечетный
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.touch_processor.set_spatial_filter_kernel(kernel_size)
            self.logger.debug(f"Размер ядра фильтра изменен: {kernel_size}")
            self._update_control_window()
    
    def _on_spatial_filter_toggle(self, value: int) -> None:
        """
        Callback для включения/выключения пространственной фильтрации
        
        Args:
            value: 0 - выключено, 1 - включено
        """
        if self.touch_processor is not None:
            enabled = bool(value)
            self.touch_processor.set_spatial_filter_enabled(enabled)
            status = "включена" if enabled else "выключена"
            self.logger.debug(f"Пространственная фильтрация {status}")
            self._update_control_window()
    
    def _on_unity_enabled_toggle(self, value: int) -> None:
        """
        Callback для включения/выключения Unity передачи
        
        Args:
            value: 0 - выключено, 1 - включено
        """
        self.unity_enabled = bool(value)
        status = "включена" if self.unity_enabled else "выключена"
        self.logger.info(f"Unity передача {status}")
        
        if self.unity_comm:
            self.unity_comm.set_enabled(self.unity_enabled)
        
        self._update_control_window()
    
    def _on_unity_port_change(self, value: int) -> None:
        """
        Callback для изменения Unity порта
        
        Args:
            value: Значение от 0 до 1000 (представляет порты от 8000 до 9000)
        """
        new_port = 8000 + value
        if new_port != self.unity_port:
            self.unity_port = new_port
            self.logger.info(f"Unity порт изменен на: {self.unity_port}")
            
            # Переинициализируем Unity коммуникацию с новым портом
            if self.unity_comm:
                self.unity_comm.set_server_address(self.unity_host, self.unity_port)
            
            self._update_control_window()
    
    def _on_advanced_filter_toggle(self, value: int) -> None:
        """
        Callback для включения/выключения продвинутой фильтрации
        
        Args:
            value: 0 - выключено, 1 - включено
        """
        if self.touch_processor is not None:
            enabled = bool(value)
            self.touch_processor.set_advanced_filter_enabled(enabled)
            status = "включена" if enabled else "выключена"
            self.logger.debug(f"Продвинутая фильтрация {status}")
            self._update_control_window()
    
    def _on_filter_threshold_change(self, value: int) -> None:
        """
        Callback для изменения порога фильтрации
        
        Args:
            value: Значение от 0 до 95 (представляет от 5 до 100 пикселей)
        """
        if self.touch_processor is not None:
            threshold = value + 5  # Конвертируем обратно: 0->5, 95->100
            self.touch_processor.set_filter_distance_threshold(threshold)
            self.logger.debug(f"Порог фильтрации изменен: {threshold}px")
            self._update_control_window()
    
    def _on_min_movement_change(self, value: int) -> None:
        """
        Callback для изменения минимального движения
        
        Args:
            value: Значение от 0 до 49 (представляет от 1 до 50 пикселей)
        """
        if self.touch_processor is not None:
            min_movement = value + 1  # Конвертируем обратно: 0->1, 49->50
            self.touch_processor.set_filter_min_movement(min_movement)
            self.logger.debug(f"Минимальное движение изменено: {min_movement}px")
            self._update_control_window()
    
    def _on_timeout_change(self, value: int) -> None:
        """
        Callback для изменения таймаута касаний
        
        Args:
            value: Значение от 0 до 99 (представляет от 0.1 до 10.0 секунд)
        """
        if self.touch_processor is not None:
            timeout = (value + 1) / 10.0  # Конвертируем обратно: 0->0.1, 99->10.0
            self.touch_processor.set_filter_timeout(timeout)
            self.logger.debug(f"Таймаут касаний изменен: {timeout}с")
            self._update_control_window()
    
    def _update_control_window(self) -> None:
        """
        Обновление информации в окне управления
        """
        if hasattr(self, 'control_window_name') and self.touch_processor is not None:
            # Создаем обновленное информационное изображение
            control_image = np.zeros((550, 500, 3), dtype=np.uint8)
            cv2.putText(control_image, "Touch Detection Settings", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        y_pos = 350  # Начинаем после существующей информации
        
        # Разделительная линия
        cv2.line(control_image, (10, y_pos - 10), (440, y_pos - 10), (100, 100, 100), 1)
        y_pos += 10
        
        # Заголовок
        cv2.putText(control_image, "Depth Filtering:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 25
        
        # Статус фильтрации
        filter_status = "ON" if self.depth_filter_enabled else "OFF"
        status_color = (0, 255, 0) if self.depth_filter_enabled else (0, 0, 255)
        cv2.putText(control_image, f"Depth Filter: {filter_status}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        y_pos += 20
        
        # Диапазон глубины
        cv2.putText(control_image, f"Depth Range: {self.min_touch_depth:.2f}-{self.max_touch_depth:.2f}m", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
        y_pos += 20
        
        # Статистика глубины (если доступна)
        if hasattr(self, '_last_cropped_depth') and self._last_cropped_depth is not None:
            depth_stats = self.touch_processor.get_depth_filter_stats(self._last_cropped_depth)
            if depth_stats and depth_stats['valid_pixels'] > 0:
                cv2.putText(control_image, f"Current Depth: {depth_stats['mean_depth']:.2f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 15
                
                cv2.putText(control_image, f"Min/Max: {depth_stats['min_depth']:.2f}m / {depth_stats['max_depth']:.2f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 15
            
            
            
            # Добавляем информацию о текущих значениях
            y_pos = 70
            line_height = 25
            
            # Основные пороги
            cv2.putText(control_image, f"Background Threshold: {self.touch_processor.get_background_threshold():.3f}m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_pos += line_height
            
            cv2.putText(control_image, f"Touch Threshold: {self.touch_processor.get_touch_threshold():.3f}m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_pos += line_height
            
            # Коррекция глубины
            cv2.putText(control_image, f"Depth Offset: {self.touch_processor.get_depth_offset():.1f}mm", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            y_pos += line_height
            
            cv2.putText(control_image, f"Depth Scale: {self.touch_processor.get_depth_scale_factor():.2f}x", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            y_pos += line_height
            
            # Фильтрация
            filter_status = "ON" if self.touch_processor.get_spatial_filter_enabled() else "OFF"
            cv2.putText(control_image, f"Spatial Filter: {filter_status} (kernel: {self.touch_processor.get_spatial_filter_kernel()})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_pos += line_height
            
            # Разделительная линия
            cv2.line(control_image, (10, y_pos + 5), (440, y_pos + 5), (100, 100, 100), 1)
            y_pos += 20
            
            # Диапазоны значений
            cv2.putText(control_image, "Ranges:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += line_height
            
            cv2.putText(control_image, "Background: -1.0m ... +3.0m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Touch: 0.0m ... 0.5m", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Depth Offset: -100mm ... +200mm", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Depth Scale: 0.8x ... 1.5x", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += 20
            
            # TouchFilter информация
            filter_status = "ON" if self.touch_processor.get_advanced_filter_enabled() else "OFF"
            cv2.putText(control_image, f"Advanced Filter: {filter_status}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            y_pos += 20
            
            cv2.putText(control_image, f"Filter Threshold: {self.touch_processor.get_filter_distance_threshold():.0f}px", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            y_pos += 20
            
            cv2.putText(control_image, f"Min Movement: {self.touch_processor.get_filter_min_movement():.0f}px", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            y_pos += 20
            
            cv2.putText(control_image, f"Touch Timeout: {self.touch_processor.get_filter_timeout():.1f}s", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            y_pos += 20
            
            # Статистика фильтра
            filter_stats = self.touch_processor.get_filter_statistics()
            cv2.putText(control_image, f"Filtered: {filter_stats['total_filtered']}/{filter_stats['total_processed']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20
            
            cv2.putText(control_image, f"Active Touches: {filter_stats['active_touches']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20
            
            cv2.putText(control_image, "Press 'ESC' to exit", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Отображаем обновленное изображение
            cv2.imshow(self.control_window_name, control_image)
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Обработчик событий мыши для калибровки поверхности
        
        Args:
            event: Тип события мыши
            x: X координата курсора
            y: Y координата курсора
            flags: Дополнительные флаги
            param: Дополнительные параметры
        """
        if (event == cv2.EVENT_LBUTTONDOWN and self.surface_calibration_mode and 
            self.touch_processor is not None and self.show_cropped):
            
            # Проверяем, что клик в области обрезанного изображения (левая часть)
            if x < 800:  # Ширина обрезанного изображения
                # Получаем текущие обрезанные изображения
                if hasattr(self, '_last_cropped_depth') and self._last_cropped_depth is not None:
                    success = self.touch_processor.add_surface_calibration_point(x, y, self._last_cropped_depth)
                    if success:
                        points_collected = len(self.touch_processor.surface_calibration_points)
                        self.logger.info(f"Точек калибровки собрано: {points_collected}/{self.surface_calibration_points_needed}")
                        
                        # Автоматическая калибровка после сбора достаточного количества точек
                        if points_collected >= self.surface_calibration_points_needed:
                            # ЗАМЕНИТЕ ЭТУ СТРОКУ:
                            # if self.touch_processor.calibrate_surface_height():
                            # НА ЭТУ:
                            if self.touch_processor.calibrate_surface_height():  # Метод уже обновлен!
                                self.surface_calibration_mode = False
                                
                                # Получаем информацию о качестве калибровки
                                quality = self.touch_processor.get_plane_calibration_quality()
                                self.logger.info(f"Калибровка поверхности завершена! Качество: {quality['quality']}")
                                self.logger.info(f"Уравнение плоскости: {quality['equation']}")
                                self.logger.info(f"Точность: RMSE={quality['rmse']:.4f}м")
                                
                                # Предупреждение если точность низкая
                                if quality['quality'] in ['fair', 'poor']:
                                    self.logger.warning("Низкая точность калибровки! Рекомендуется перекалибровать поверхность.")
                            else:
                                self.logger.warning("Не удалось завершить калибровку поверхности")
                else:
                    self.logger.warning("Нет данных о глубине для калибровки")
    
    
    def _on_depth_filter_toggle(self, value: int) -> None:
        """
        Callback для включения/выключения фильтрации по глубине
        
        Args:
            value: 0 - выключено, 1 - включено
        """
        self.depth_filter_enabled = bool(value)
        status = "включена" if self.depth_filter_enabled else "выключена"
        self.logger.info(f"Фильтрация по глубине {status}")
        
        # Применяем настройки к TouchProcessor
        if self.touch_processor is not None:
            self.touch_processor.enable_depth_filter(self.depth_filter_enabled)
            self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
        
        self._update_control_window()

    def _on_min_depth_change(self, value: int) -> None:
        """
        Callback для изменения минимальной глубины
        
        Args:
            value: Значение от 0 до 1900 (представляет от 100мм до 2000мм)
        """
        # Конвертируем обратно в метры: 0->0.1м, 1000->1.1м, 1900->2.0м
        min_depth_meters = (value + 100) / 1000.0
        
        # Проверяем, чтобы минимальная глубина была меньше максимальной
        if min_depth_meters < self.max_touch_depth - 0.1:  # Минимум на 10см меньше
            self.min_touch_depth = min_depth_meters
            self.logger.debug(f"Минимальная глубина изменена: {min_depth_meters:.2f}м")
            
            # Применяем настройки к TouchProcessor
            if self.touch_processor is not None:
                self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
            
            self._update_control_window()
        else:
            self.logger.warning("Минимальная глубина должна быть меньше максимальной хотя бы на 10см")

    def _on_max_depth_change(self, value: int) -> None:
        """
        Callback для изменения максимальной глубины
        
        Args:
            value: Значение от 0 до 4500 (представляет от 500мм до 5000мм)
        """
        # Конвертируем обратно в метры: 0->0.5м, 1000->1.5м, 4500->5.0м
        max_depth_meters = (value + 500) / 1000.0
        
        # Проверяем, чтобы максимальная глубина была больше минимальной
        if max_depth_meters > self.min_touch_depth + 0.1:  # Минимум на 10см больше
            self.max_touch_depth = max_depth_meters
            self.logger.debug(f"Максимальная глубина изменена: {max_depth_meters:.2f}м")
            
            # Применяем настройки к TouchProcessor
            if self.touch_processor is not None:
                self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
            
            self._update_control_window()
        else:
            self.logger.warning("Максимальная глубина должна быть больше минимальной хотя бы на 10см")
            
    def test_camera_connection(self) -> bool:
        """
        Тест подключения камеры RealSense
        
        Returns:
            True если камера доступна, False в противном случае
        """
        try:
            test_camera: RealSenseCamera = RealSenseCamera()
            if test_camera.start():
                self.logger.info("Тест подключения камеры прошел успешно")
                test_camera.stop()
                return True
            else:
                self.logger.error("Не удалось подключиться к камере")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при тестировании камеры: {e}")
            return False
    
    def _create_display_image(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Создание комбинированного изображения для отображения
        
        Args:
            color_image: Цветное изображение
            depth_image: Изображение глубины
            
        Returns:
            Комбинированное изображение
        """
        if self.show_cropped and self.cropper.is_calibrated:
            # Режим отображения обрезанных изображений
            cropped_color, cropped_depth = self.cropper.crop_both_images(color_image, depth_image)
            
            if cropped_color is not None and cropped_depth is not None:
                # Применяем цветовую карту к обрезанному изображению глубины
                depth_colormap: np.ndarray = self.camera.apply_colormap_to_depth(cropped_depth)
                
                # Создаем комбинированное изображение
                if self.show_depth:
                    images: np.ndarray = np.hstack((cropped_color, depth_colormap))
                else:
                    images: np.ndarray = np.hstack((cropped_color))
                
                # Добавляем текст с информацией
                cv2.putText(
                    images,
                    'Cropped Color',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                if self.show_depth:
                    cv2.putText(
                        images,
                        'Cropped Depth',
                        (cropped_color.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                # Добавляем информацию о размере
                cv2.putText(
                    images,
                    f'Size: {cropped_color.shape[1]}x{cropped_color.shape[0]}',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                                # Информация о фильтрации по глубине
                depth_filter_status = "ON" if self.depth_filter_enabled else "OFF"
                depth_filter_color = (0, 255, 0) if self.depth_filter_enabled else (0, 0, 255)
                cv2.putText(
                    images,
                    f'Depth Filter: {depth_filter_status} ({self.min_touch_depth:.1f}-{self.max_touch_depth:.1f}m)',
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    depth_filter_color,
                    1
                )
                
                # Сохраняем последние обрезанные изображения для калибровки поверхности
                self._last_cropped_depth = cropped_depth
                
                # Обрабатываем касания если включена детекция
                if self.touch_detection_enabled and self.touch_processor is not None:
                    self.current_touches = self.touch_processor.process_frame(cropped_color, cropped_depth)
                    
                    # Отправляем касания в Unity если включена передача
                    if self.unity_enabled and self.unity_comm is not None and self.current_touches:
                        self.unity_comm.send_touch_coordinates(self.current_touches)
                    
                    # Добавляем отладочные изображения если включен режим отладки
                    if self.show_debug_images:
                        debug_images = self.touch_processor.get_debug_images()
                        debug_list = []
                        
                        # Добавляем бинарное изображение зоны детекции
                        if debug_images['binary'] is not None:
                            binary_colored = cv2.applyColorMap(debug_images['binary'], cv2.COLORMAP_HOT)
                            # Добавляем подпись
                            cv2.putText(
                                binary_colored,
                                'Binary Detection',
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )
                            debug_list.append(binary_colored)
                        
                        # Добавляем нормализованное разностное изображение
                        if debug_images['depth_diff_normalized'] is not None:
                            diff_colored = cv2.applyColorMap(debug_images['depth_diff_normalized'], cv2.COLORMAP_VIRIDIS)
                            # Добавляем подпись
                            cv2.putText(
                                diff_colored,
                                'Depth Difference',
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )
                            debug_list.append(diff_colored)
                        
                        # Объединяем с основными изображениями
                        if debug_list:
                            images = np.hstack([images] + debug_list)
                    
                    # Визуализируем касания на обрезанном изображении
                    if self.show_touches and self.current_touches:
                        images = self._visualize_touches_on_cropped(images, cropped_color)
                
                # Визуализация режима калибровки поверхности
                if self.surface_calibration_mode and self.touch_processor is not None:
                    images = self._visualize_surface_calibration(images)
                
                # Добавляем информацию о касаниях и режиме отладки
                touch_status = "ON" if self.touch_detection_enabled else "OFF"
                cv2.putText(
                    images,
                    f'Touch Detection: {touch_status}',
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if self.touch_detection_enabled else (0, 0, 255),
                    1
                )
                
                # Информация о режиме отладки
                debug_status = "ON" if self.show_debug_images else "OFF"
                cv2.putText(
                    images,
                    f'Debug Mode: {debug_status}',
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0) if self.show_debug_images else (128, 128, 128),
                    1
                )
                
                if self.current_touches:
                    cv2.putText(
                        images,
                        f'Touches: {len(self.current_touches)}',
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1
                    )
                
                # Добавляем информацию о чувствительности
                if self.touch_processor is not None:
                    sensitivity_text = f'Sensitivity: {self.touch_processor.sensitivity_level}/10'
                    cv2.putText(
                        images,
                        sensitivity_text,
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1
                    )
                    
                    # Информация о калибровке поверхности
                    surface_status = "Calibrated" if self.touch_processor.surface_height is not None else "Not Calibrated"
                    surface_color = (0, 255, 0) if self.touch_processor.surface_height is not None else (0, 0, 255)
                    cv2.putText(
                        images,
                        f'Surface: {surface_status}',
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        surface_color,
                        1
                    )
            else:
                # Если обрезка не удалась, показываем исходные изображения
                return self._create_original_display_image(color_image, depth_image)
        else:
            # Режим отображения исходных изображений
            images = self._create_original_display_image(color_image, depth_image)
        
        # Добавляем инструкции по управлению
        if self.surface_calibration_mode:
            instructions = [
                "SURFACE CALIBRATION MODE - Click on surface points",
                f"Points collected: {len(self.touch_processor.surface_calibration_points) if self.touch_processor else 0}/{self.surface_calibration_points_needed}",
                "ESC - exit surface calibration",
                "Q - exit program"
            ]
        else:
            # В секции инструкций добавьте новые команды:
            instructions = [
                "C - enter calibration mode",
                "V - toggle cropped/original view",
                "P - toggle projection area overlay", 
                "T - toggle touch detection",
                "S - enter surface calibration mode",
                "D - toggle debug images (binary detection)",
                "+ / - - adjust touch sensitivity",
                "B - set background (when touch detection on)",
                "R - reset touch background",
                "F - toggle depth filtering",  # НОВАЯ КОМАНДА
                "1/2 - increase/decrease min depth",  # НОВАЯ КОМАНДА
                "3/4 - increase/decrease max depth",  # НОВАЯ КОМАНДА
                "5 - auto set depth range",  # НОВАЯ КОМАНДА
                "Q - exit program"
            ]
        
        y_offset = images.shape[0]//2 - 100
        for instruction in instructions:
            cv2.putText(
                images,
                instruction,
                (8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Черная обводка
                3,  # Толщина обводки
                cv2.LINE_AA
            )
            cv2.putText(
                images,
                instruction,
                (8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
            y_offset += 20
        
        return images
    
    def _create_original_display_image(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Создание комбинированного изображения с исходными кадрами
        
        Args:
            color_image: Цветное изображение
            depth_image: Изображение глубины
            
        Returns:
            Комбинированное изображение
        """
        # Показываем область проекции если нужно и есть калибровка
        if self.show_projection_area and self.cropper.is_calibrated:
            color_with_overlay = self.cropper.visualize_crop_area(color_image)
        else:
            color_with_overlay = color_image.copy()
        
        # Применяем цветовую карту к изображению глубины для визуализации
        depth_colormap: np.ndarray = self.camera.apply_colormap_to_depth(depth_image)
        
        # Создаем комбинированное изображение для отображения
        images: np.ndarray = np.hstack((color_with_overlay, depth_colormap))
        
        # Добавляем текст с информацией
        cv2.putText(
            images,
            'Original Color',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            images,
            'Original Depth',
            (color_image.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Добавляем статус калибровки
        calibration_status = "Calibrated" if self.cropper.is_calibrated else "Not Calibrated"
        cv2.putText(
            images,
            f'Status: {calibration_status}',
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if self.cropper.is_calibrated else (0, 0, 255),
            1
        )
        
        return images
    
    def _visualize_touches_on_cropped(self, combined_image: np.ndarray, cropped_color: np.ndarray) -> np.ndarray:
        """
        Визуализация касаний на комбинированном изображении с обрезанными кадрами
        
        Args:
            combined_image: Комбинированное изображение (color + depth)
            cropped_color: Обрезанное цветное изображение
            
        Returns:
            Изображение с визуализированными касаниями
        """
        result = combined_image.copy()
        
        for i, (scaled_x, scaled_y, touch_info) in enumerate(self.current_touches):
            # Получаем исходные координаты в кропнутом изображении
            orig_x = touch_info["original_x"]
            orig_y = touch_info["original_y"]
            confidence = touch_info["confidence"]
            area = touch_info["area"]
            
            # Цвет зависит от уверенности
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            # Рисуем круг в точке касания на левой (цветной) части
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
                2
            )
            
            # Добавляем информацию о касании - координаты в целевом разрешении
            info_text = f"({scaled_x},{scaled_y})"
            cv2.putText(
                result,
                info_text,
                (orig_x + 20, orig_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
            # Показываем также на depth части (правая часть изображения)
            depth_x = orig_x + cropped_color.shape[1]
            cv2.circle(result, (depth_x, orig_y), 8, color, 2)
            
            # Выводим информацию о касании в консоль
            self.logger.info(f"Touch {i+1}: screen({scaled_x},{scaled_y}) crop({orig_x},{orig_y}) conf={confidence:.2f}")
        
        return result
    
    def _visualize_surface_calibration(self, combined_image: np.ndarray) -> np.ndarray:
        """
        Визуализация режима калибровки поверхности
        
        Args:
            combined_image: Комбинированное изображение
            
        Returns:
            Изображение с визуализацией калибровки поверхности
        """
        result = combined_image.copy()
        
        if self.touch_processor is not None:
            # Отображаем уже собранные точки калибровки
            for i, (x, y, depth) in enumerate(self.touch_processor.surface_calibration_points):
                cv2.circle(result, (x, y), 8, (255, 0, 255), -1)  # Фиолетовые точки
                cv2.circle(result, (x, y), 12, (255, 0, 255), 2)
                
                # Номер точки
                cv2.putText(
                    result,
                    str(i + 1),
                    (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2
                )
            
            # Добавляем информацию о режиме калибровки
            cv2.putText(
                result,
                "SURFACE CALIBRATION MODE",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )
            
            points_text = f"Points: {len(self.touch_processor.surface_calibration_points)}/{self.surface_calibration_points_needed}"
            cv2.putText(
                result,
                points_text,
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )
        
        return result
    
    def _enter_calibration_mode(self) -> None:
        """
        Вход в режим калибровки проекции
        """
        self.logger.info("Переход в режим калибровки проекции")
        self.is_calibration_mode = True
        
        # Создаем калибратор
        self.calibrator = ProjectionCalibrator(self.camera)
        
        # Закрываем основное окно
        cv2.destroyWindow(self.window_name)
        
        # Запускаем калибровку
        calibration_success = self.calibrator.calibrate()
        
        if calibration_success:
            self.logger.info("Калибровка завершена успешно")
            # Перезагружаем cropper с новыми данными калибровки
            self.cropper = ProjectionAreaCropper(output_size=(800, 600))
            if self.cropper.is_calibrated:
                self.logger.info("Обрезчик проекции обновлен с новой калибровкой")
                # Переинициализируем процессор касаний
                self._init_touch_processor()
            else:
                self.logger.warning("Не удалось загрузить новые данные калибровки в обрезчик")
        else:
            self.logger.info("Калибровка была отменена")
        
        # Возвращаемся в основной режим
        self._exit_calibration_mode()
    
    def _exit_calibration_mode(self) -> None:
        """
        Выход из режима калибровки
        """
        self.is_calibration_mode = False
        self.calibrator = None
        self.logger.info("Возврат в основной режим")
    
    def run(self) -> None:
        """
        Запуск основного цикла детекции касаний
        """
        try:
            # Используем контекстный менеджер для автоматического управления ресурсами
            with self.camera:
                self.logger.info("Камера запущена. Нажмите 'q' для выхода")
                
                # Получаем масштаб глубины
                depth_scale: float = self.camera.get_depth_scale()
                self.logger.info(f"Масштаб глубины: {depth_scale:.6f}")
                
                # Получаем внутренние параметры камеры
                color_intrinsics, depth_intrinsics = self.camera.get_intrinsics()
                if color_intrinsics and depth_intrinsics:
                    self.logger.info(f"Внутренние параметры цветной камеры: {color_intrinsics.width}x{color_intrinsics.height}")
                    self.logger.info(f"Внутренние параметры камеры глубины: {depth_intrinsics.width}x{depth_intrinsics.height}")
                
                # Основной цикл обработки
                self._main_loop()
                
        except RuntimeError as e:
            self.logger.error(f"Ошибка инициализации камеры: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.info("Прерывание от пользователя")
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка: {e}")
            sys.exit(1)
        finally:
            # Закрываем Unity коммуникацию
            if self.unity_comm:
                self.unity_comm.close()
            
            # Закрываем все окна OpenCV
            cv2.destroyAllWindows()
            self.logger.info("Приложение завершено")
    
    def _main_loop(self) -> None:
        """
        Основной цикл обработки кадров
        """
        while True:
            # Если мы в режиме калибровки, пропускаем обработку основного цикла
            if self.is_calibration_mode:
                continue
            
            # Получаем кадры с камеры
            color_image, depth_image = self.camera.get_frames()
            
            if color_image is None or depth_image is None:
                self.logger.warning("Не удалось получить кадры с камеры")
                continue
            
            # Создаем изображение для отображения
            display_image = self._create_display_image(color_image, depth_image)
            
            # Отображаем изображения
            cv2.imshow(self.window_name, display_image)
            # Обработка нажатий клавиш
            key: int = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                # НОВЫЕ ОБРАБОТЧИКИ ДЛЯ ФИЛЬТРАЦИИ ПО ГЛУБИНЕ
            elif key == ord('f'):  # Включение/выключение фильтрации по глубине
                self.depth_filter_enabled = not self.depth_filter_enabled
                status = "включена" if self.depth_filter_enabled else "выключена"
                self.logger.info(f"Фильтрация по глубине {status}")
                
                if self.touch_processor is not None:
                    self.touch_processor.enable_depth_filter(self.depth_filter_enabled)
            
            elif key == ord('1'):  # Увеличить минимальную глубину
                if self.touch_processor is not None:
                    new_min = min(self.max_touch_depth - 0.1, self.min_touch_depth + 0.1)
                    if new_min != self.min_touch_depth:
                        self.min_touch_depth = new_min
                        self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
                        self.logger.info(f"Минимальная глубина увеличена до: {self.min_touch_depth:.2f}м")
            
            elif key == ord('2'):  # Уменьшить минимальную глубину
                if self.touch_processor is not None:
                    new_min = max(0.1, self.min_touch_depth - 0.1)
                    if new_min != self.min_touch_depth:
                        self.min_touch_depth = new_min
                        self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
                        self.logger.info(f"Минимальная глубина уменьшена до: {self.min_touch_depth:.2f}м")
            
            elif key == ord('3'):  # Увеличить максимальную глубину
                if self.touch_processor is not None:
                    new_max = min(5.0, self.max_touch_depth + 0.1)
                    if new_max != self.max_touch_depth:
                        self.max_touch_depth = new_max
                        self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
                        self.logger.info(f"Максимальная глубина увеличена до: {self.max_touch_depth:.2f}м")
            
            elif key == ord('4'):  # Уменьшить максимальную глубину
                if self.touch_processor is not None:
                    new_max = max(self.min_touch_depth + 0.1, self.max_touch_depth - 0.1)
                    if new_max != self.max_touch_depth:
                        self.max_touch_depth = new_max
                        self.touch_processor.set_depth_filter_range(self.min_touch_depth, self.max_touch_depth)
                        self.logger.info(f"Максимальная глубина уменьшена до: {self.max_touch_depth:.2f}м")
            
            elif key == ord('5'):  # Автоматическая настройка диапазона глубины
                if (self.touch_processor is not None and 
                    hasattr(self.touch_processor, 'plane_A') and 
                    self.touch_processor.plane_A is not None):
                    
                    # Автоматическая настройка на основе калиброванной поверхности
                    if hasattr(self, '_last_cropped_depth') and self._last_cropped_depth is not None:
                        self.touch_processor.auto_set_depth_range(self._last_cropped_depth, margin=0.2)
                        
                        # Обновляем наши переменные
                        self.min_touch_depth = self.touch_processor.min_touch_depth
                        self.max_touch_depth = self.touch_processor.max_touch_depth
                        self.logger.info(f"Автоматически установлен диапазон глубины: {self.min_touch_depth:.2f}-{self.max_touch_depth:.2f}м")
                else:
                    self.logger.warning("Для автоматической настройки нужна калиброванная поверхность")
            elif key == ord('c'):
                # Переход в режим калибровки
                self._enter_calibration_mode()
            elif key == ord('v'):
                # Переключение между исходным и обрезанным видом
                if self.cropper.is_calibrated:
                    self.show_cropped = not self.show_cropped
                    mode = "cropped" if self.show_cropped else "original"
                    self.logger.info(f"Переключен режим отображения: {mode}")
                else:
                    self.logger.warning("Сначала выполните калибровку для работы с обрезанными изображениями")
            elif key == ord('p'):
                # Переключение отображения области проекции
                self.show_projection_area = not self.show_projection_area
                status = "включено" if self.show_projection_area else "выключено"
                self.logger.info(f"Отображение области проекции: {status}")
            elif key == ord('t'):
                # Переключение детекции касаний
                if self.touch_processor is not None:
                    self.touch_detection_enabled = not self.touch_detection_enabled
                    status = "включена" if self.touch_detection_enabled else "выключена"
                    self.logger.info(f"Детекция касаний: {status}")
                    if not self.touch_detection_enabled:
                        self.current_touches.clear()
                else:
                    self.logger.warning("TouchProcessor не инициализирован. Сначала выполните калибровку.")
            elif key == ord('b'):
                # Установка фонового изображения для детекции касаний
                if self.touch_detection_enabled and self.touch_processor is not None and self.show_cropped:
                    # Получаем текущие обрезанные изображения
                    cropped_color, cropped_depth = self.cropper.crop_both_images(color_image, depth_image)
                    if cropped_depth is not None:
                        self.touch_processor.set_background(cropped_depth)
                        self.logger.info("Фоновое изображение для детекции касаний установлено")
                    else:
                        self.logger.warning("Не удалось получить обрезанное изображение глубины")
                else:
                    self.logger.warning("Для установки фона включите детекцию касаний и режим обрезанного вида")
            elif key == ord('r'):
                # Сброс фонового изображения
                if self.touch_processor is not None:
                    self.touch_processor.reset_background()
                    self.current_touches.clear()
                    self.logger.info("Фоновое изображение сброшено")
                else:
                    self.logger.warning("TouchProcessor не инициализирован")
            elif key == ord('d'):
                # Переключение режима отладки
                self.show_debug_images = not self.show_debug_images
                status = "ВКЛЮЧЕН" if self.show_debug_images else "ВЫКЛЮЧЕН"
                self.logger.info(f"Режим отладки {status}")
                
                if self.show_debug_images:
                    self.logger.info("Показывать отладочные изображения: бинарная маска детекции и разность глубины")
                    if not self.touch_detection_enabled:
                        self.logger.warning("Для отображения отладочных изображений включите детекцию касаний (T)")
                    if not self.show_cropped:
                        self.logger.warning("Для отображения отладочных изображений переключитесь в режим кропнутого вида (V)")
            elif key == ord('s'):
                # Вход в режим калибровки поверхности
                if self.touch_processor is not None and self.show_cropped:
                    if not self.surface_calibration_mode:
                        self.surface_calibration_mode = True
                        self.touch_processor.clear_surface_calibration()
                        self.logger.info("Начата калибровка поверхности. Кликните на 5 точек поверхности проекции.")
                    else:
                        # Завершаем калибровку вручную
                        if len(self.touch_processor.surface_calibration_points) >= 3:
                            if self.touch_processor.calibrate_surface_height():
                                self.surface_calibration_mode = False
                                self.logger.info("Калибровка поверхности завершена вручную")
                            else:
                                self.logger.warning("Не удалось завершить калибровку поверхности")
                        else:
                            self.logger.warning("Недостаточно точек для калибровки (минимум 3)")
                else:
                    self.logger.warning("Для калибровки поверхности включите обрезанный вид и инициализируйте TouchProcessor")
            elif key == ord('+') or key == ord('='):
                # Увеличение чувствительности
                if self.touch_processor is not None:
                    self.touch_processor.adjust_sensitivity(1)
                else:
                    self.logger.warning("TouchProcessor не инициализирован")
            elif key == ord('-'):
                # Уменьшение чувствительности
                if self.touch_processor is not None:
                    self.touch_processor.adjust_sensitivity(-1)
                else:
                    self.logger.warning("TouchProcessor не инициализирован")
            elif key == 27:  # ESC
                # Выход из режима калибровки поверхности
                if self.surface_calibration_mode:
                    self.surface_calibration_mode = False
                    self.logger.info("Калибровка поверхности отменена")


def main() -> None:
    """
    Основная функция для запуска системы детекции касаний
    """
    # Создаем экземпляр камеры
    camera: RealSenseCamera = RealSenseCamera(width=640, height=480, fps=15)
    
    # Создаем детектор касаний
    touch_detector: TouchDetector = TouchDetector(camera)
    
    # Сначала проверим подключение камеры
    if touch_detector.test_camera_connection():
        touch_detector.run()
    else:
        print("Не удалось подключиться к камере RealSense D435F")
        print("Убедитесь что:")
        print("1. Камера подключена к USB 3.0 порту")
        print("2. Установлены драйверы Intel RealSense")
        print("3. Установлена библиотека pyrealsense2")
        sys.exit(1)


if __name__ == "__main__":
    main()
