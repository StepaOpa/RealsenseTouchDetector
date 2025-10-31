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

import json
import os


class TouchDetector:
    """
    Класс для детекции касаний на проекции
    """

    def __init__(self, camera: RealSenseCamera) -> None:
        """
        Инициализация детектора касаний
        Args:
            camera: Экземпляр камеры RealSense
            settings_file: Путь к файлу настроек (по умолчанию "touch_detector_settings.json")
        """
        self.camera: RealSenseCamera = camera
        self.settings_file: str = settings_file  # Сохраняем имя файла настроек
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.window_name: str = "RealSense D435F - Touch Detection"
        self.camera_width: int = 1280
        self.camera_height: int = 720
        self.crop_width: int = 1024
        self.crop_height: int = 768

        # --- НАЧАЛО: Инициализация параметров с значениями по умолчанию ---
        # Эти значения будут перезаписаны при загрузке из файла, если он существует
        self.depth_filter_enabled: bool = True  # Включена ли фильтрация по глубине
        self.min_touch_depth: float = (
            0.3  # Минимальная глубина касания в метрах (30 см)
        )
        self.max_touch_depth: float = (
            2.0  # Максимальная глубина касания в метрах (2 метра)
        )
        self.show_cropped: bool = (
            False  # Режим отображения: False - исходные, True - обрезанные
        )
        self.show_projection_area: bool = (
            True  # Показывать ли область проекции на исходном изображении
        )
        self.touch_detection_enabled: bool = False  # Включена ли детекция касаний
        self.show_touches: bool = True  # Показывать ли касания на изображении
        self.show_debug_images: bool = (
            False  # Показывать ли отладочные изображения детекции
        )
        self.unity_enabled: bool = True  # Включена ли отправка в Unity
        self.unity_host: str = "127.0.0.1"  # IP адрес Unity
        self.unity_port: int = 8052  # UDP порт для Unity
        self.surface_calibration_mode: bool = (
            False  # Режим калибровки поверхности (обычно False при запуске)
        )
        self.show_depth: bool = True

        # Параметры, хранящиеся в TouchProcessor (если он инициализирован)
        # Эти значения по умолчанию будут установлены в TouchProcessor при его инициализации
        # или загружены из настроек и переданы ему.
        self.default_touch_offset_x: int = 0
        self.default_touch_offset_y: int = 0
        self.default_surface_filter_enabled: bool = False
        self.default_surface_min_offset: float = -0.1  # в метрах
        self.default_surface_max_offset: float = 0.1  # в метрах
        self.default_flip_180: bool = False
        self.default_depth_offset: int = 0  # в мм
        self.default_depth_scale_factor: float = 1.0
        self.default_unity_send_enabled: bool = True
        self.default_min_depth_mm: int = 300  # в мм (0.3м)
        self.default_max_depth_mm: int = 2000  # в мм (2.0м)
        self.default_min_touch_area: int = 100
        self.default_max_touch_area: int = 5000
        # --- КОНЕЦ: Инициализация параметров ---

        # Калибратор проекции
        self.calibrator: Optional[ProjectionCalibrator] = None
        self.is_calibration_mode: bool = False

        # Обрезчик проекции
        self.cropper: ProjectionAreaCropper = ProjectionAreaCropper(
            output_size=(self.crop_width, self.crop_height)
        )

        # Процессор касаний
        self.touch_processor: Optional[TouchProcessor] = None
        self.current_touches: List = []  # Текущие обнаруженные касания

        # Unity коммуникация
        self.unity_comm: Optional[UnityCommunication] = None

        # Калибровка поверхности
        self.surface_calibration_points_needed: int = (
            5  # Количество точек для калибровки
        )

        # --- ЗАГРУЗКА НАСТРОЕК ---
        self._load_settings()

        # Инициализация камеры
        self._setup_logging()

        # Инициализируем процессор касаний если есть калибровка
        # (Это может изменить параметры, хранящиеся в touch_processor)
        self._init_touch_processor()

        # Настройка обработчика мыши
        self._setup_mouse_callback()

        # Создание окна управления с ползунками
        # (Ползунки будут установлены в значения, загруженные из файла)
        self._setup_control_window()

        # --- МЕТОД ЗАГРУЗКИ НАСТРОЕК ---

    def _load_settings(self) -> None:
        """
        Загружает настройки из JSON файла.
        """
        if not os.path.exists(self.settings_file):
            self.logger.info(
                f"Файл настроек {self.settings_file} не найден. Используются значения по умолчанию."
            )
            return

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            # Загружаем параметры, хранящиеся в TouchDetector
            self.depth_filter_enabled = saved_data.get(
                "depth_filter_enabled", self.depth_filter_enabled
            )
            self.min_touch_depth = saved_data.get(
                "min_touch_depth", self.min_touch_depth
            )
            self.max_touch_depth = saved_data.get(
                "max_touch_depth", self.max_touch_depth
            )
            self.show_cropped = saved_data.get("show_cropped", self.show_cropped)
            self.show_projection_area = saved_data.get(
                "show_projection_area", self.show_projection_area
            )
            self.touch_detection_enabled = saved_data.get(
                "touch_detection_enabled", self.touch_detection_enabled
            )
            self.show_touches = saved_data.get("show_touches", self.show_touches)
            self.show_debug_images = saved_data.get(
                "show_debug_images", self.show_debug_images
            )
            self.unity_enabled = saved_data.get("unity_enabled", self.unity_enabled)
            self.unity_host = saved_data.get("unity_host", self.unity_host)
            self.unity_port = saved_data.get("unity_port", self.unity_port)
            self.show_depth = saved_data.get("show_depth", self.show_depth)
            # Загружаем параметры, которые будут применены к TouchProcessor при его инициализации
            self.default_touch_offset_x = saved_data.get(
                "default_touch_offset_x", self.default_touch_offset_x
            )
            self.default_touch_offset_y = saved_data.get(
                "default_touch_offset_y", self.default_touch_offset_y
            )
            self.default_surface_filter_enabled = saved_data.get(
                "default_surface_filter_enabled", self.default_surface_filter_enabled
            )
            self.default_surface_min_offset = saved_data.get(
                "default_surface_min_offset", self.default_surface_min_offset
            )
            self.default_surface_max_offset = saved_data.get(
                "default_surface_max_offset", self.default_surface_max_offset
            )
            self.default_flip_180 = saved_data.get(
                "default_flip_180", self.default_flip_180
            )
            self.default_depth_offset = saved_data.get(
                "default_depth_offset", self.default_depth_offset
            )
            self.default_depth_scale_factor = saved_data.get(
                "default_depth_scale_factor", self.default_depth_scale_factor
            )
            self.default_unity_send_enabled = saved_data.get(
                "default_unity_send_enabled", self.default_unity_send_enabled
            )
            self.default_min_depth_mm = saved_data.get(
                "default_min_depth_mm", self.default_min_depth_mm
            )
            self.default_max_depth_mm = saved_data.get(
                "default_max_depth_mm", self.default_max_depth_mm
            )
            self.default_min_touch_area = saved_data.get(
                "default_min_touch_area", self.default_min_touch_area
            )
            self.default_max_touch_area = saved_data.get(
                "default_max_touch_area", self.default_max_touch_area
            )

            self.logger.info(f"Настройки загружены из {self.settings_file}")
        except Exception as e:
            self.logger.error(
                f"Ошибка загрузки настроек из {self.settings_file}: {e}. Используются значения по умолчанию."
            )

    # --- МЕТОД СОХРАНЕНИЯ НАСТРОЕК ---
    def _save_settings(self) -> None:
        """
        Сохраняет текущие настройки в JSON файл.
        """
        settings_to_save = {
            "depth_filter_enabled": self.depth_filter_enabled,
            "min_touch_depth": self.min_touch_depth,
            "max_touch_depth": self.max_touch_depth,
            "show_cropped": self.show_cropped,
            "show_projection_area": self.show_projection_area,
            "touch_detection_enabled": self.touch_detection_enabled,
            "show_touches": self.show_touches,
            "show_debug_images": self.show_debug_images,
            "unity_enabled": self.unity_enabled,
            "unity_host": self.unity_host,
            "unity_port": self.unity_port,
            "show_depth": self.show_depth,
            # Параметры, которые хранятся в TouchProcessor (если он инициализирован)
            # Сохраняем текущие значения из TouchProcessor, если он есть, иначе из default_*
            "default_touch_offset_x": (
                self.touch_processor.touch_offset_x
                if self.touch_processor
                else self.default_touch_offset_x
            ),
            "default_touch_offset_y": (
                self.touch_processor.touch_offset_y
                if self.touch_processor
                else self.default_touch_offset_y
            ),
            "default_surface_filter_enabled": (
                self.touch_processor.is_surface_detection_enabled()
                if self.touch_processor
                else self.default_surface_filter_enabled
            ),
            "default_surface_min_offset": (
                self.touch_processor.surface_min_offset
                if self.touch_processor
                else self.default_surface_min_offset
            ),
            "default_surface_max_offset": (
                self.touch_processor.surface_max_offset
                if self.touch_processor
                else self.default_surface_max_offset
            ),
            "default_flip_180": (
                self.touch_processor.is_flip_180_enabled()
                if self.touch_processor
                else self.default_flip_180
            ),
            "default_depth_offset": (
                self.touch_processor.get_depth_offset()
                if self.touch_processor
                else self.default_depth_offset
            ),
            "default_depth_scale_factor": (
                self.touch_processor.get_depth_scale_factor()
                if self.touch_processor
                else self.default_depth_scale_factor
            ),
            "default_unity_send_enabled": self.unity_enabled,  # или отдельный флаг в TouchProcessor, если есть
            "default_min_depth_mm": int(
                self.min_touch_depth * 1000
            ),  # Сохраняем как мм
            "default_max_depth_mm": int(
                self.max_touch_depth * 1000
            ),  # Сохраняем как мм
            "default_min_touch_area": (
                self.touch_processor.get_min_touch_area()
                if self.touch_processor
                else self.default_min_touch_area
            ),
            "default_max_touch_area": (
                self.touch_processor.get_max_touch_area()
                if self.touch_processor
                else self.default_max_touch_area
            ),
        }

        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings_to_save, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Настройки сохранены в {self.settings_file}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения настроек в {self.settings_file}: {e}")

    def _setup_logging(self) -> None:
        """
        Настройка логирования
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _init_touch_processor(self) -> None:
        """
        Инициализация процессора касаний
        """
        if self.cropper.is_calibrated:
            self.touch_processor = TouchProcessor(
                cropped_width=self.crop_width,
                cropped_height=self.crop_height,
                target_width=1920,
                target_height=1080,
            )
            self.logger.info("TouchProcessor инициализирован")

            # --- ПРИМЕНЕНИЕ ЗАГРУЖЕННЫХ НАСТРОЕК К TouchProcessor ---
            # Эти параметры устанавливаются после инициализации
            if self.touch_processor:
                self.touch_processor.touch_offset_x = self.default_touch_offset_x
                self.touch_processor.touch_offset_y = self.default_touch_offset_y
                self.touch_processor.enable_surface_detection(
                    self.default_surface_filter_enabled
                )
                self.touch_processor.set_surface_detection_range(
                    self.default_surface_min_offset, self.default_surface_max_offset
                )
                self.touch_processor.enable_flip_180(self.default_flip_180)
                self.touch_processor.set_depth_offset(self.default_depth_offset)
                self.touch_processor.set_depth_scale_factor(
                    self.default_depth_scale_factor
                )
                # Устанавливаем Unity send включенным, если в настройках было так
                if hasattr(self.touch_processor, "set_unity_send_enabled"):
                    self.touch_processor.set_unity_send_enabled(
                        self.default_unity_send_enabled
                    )
                self.touch_processor.set_min_touch_area(self.default_min_touch_area)
                self.touch_processor.set_max_touch_area(self.default_max_touch_area)

            # НАСТРАИВАЕМ ФИЛЬТРАЦИЮ ПО ГЛУБИНЕ (теперь с загруженными значениями)
            self.touch_processor.enable_depth_filter(self.depth_filter_enabled)
            self.touch_processor.set_depth_filter_range(
                self.min_touch_depth, self.max_touch_depth
            )
            self.logger.info(
                f"Фильтрация по глубине: {'включена' if self.depth_filter_enabled else 'выключена'}"
            )
            self.logger.info(
                f"Диапазон глубины: {self.min_touch_depth:.2f}-{self.max_touch_depth:.2f}м"
            )
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
                target_height=1080,
            )

            # Тестируем соединение
            if self.unity_comm.test_connection():
                self.logger.info(
                    f"Unity коммуникация инициализирована: {self.unity_host}:{self.unity_port}"
                )
            else:
                self.logger.warning(
                    "Unity приложение не отвечает, но модуль готов к работе"
                )

        except Exception as e:
            self.logger.error(f"Ошибка инициализации Unity коммуникации: {e}")
            self.unity_comm = None

    def _setup_mouse_callback(self) -> None:
        """
        Настройка обработчика мыши для калибровки поверхности
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def _on_touch_offset_x_change(self, value: int) -> None:
        if self.touch_processor is not None:
            offset_x = value - 500  # обратно в диапазон [-50, +50]
            self.touch_processor.touch_offset_x = offset_x
            self.logger.debug(f"Смещение по X: {offset_x}")
            self._update_control_window()

    def _on_touch_offset_y_change(self, value: int) -> None:
        if self.touch_processor is not None:
            offset_y = value - 500
            self.touch_processor.touch_offset_y = offset_y
            self.logger.debug(f"Смещение по Y: {offset_y}")
            self._update_control_window()

    def _on_surface_filter_toggle(self, value: int) -> None:
        if self.touch_processor is not None:
            enabled = bool(value)
            self.touch_processor.enable_surface_detection(enabled)
            self._update_control_window()

    def _on_surface_min_offset_change(self, value: int) -> None:
        if self.touch_processor is not None:
            min_offset = (value - 1000) / 1000.0  # из мм в метры: 0→-1.0, 2000→+1.0
            _, current_max = self.touch_processor.get_surface_detection_range()
            if min_offset < current_max:
                self.touch_processor.set_surface_detection_range(
                    min_offset, current_max
                )
            self._update_control_window()

    def _on_surface_max_offset_change(self, value: int) -> None:
        if self.touch_processor is not None:
            max_offset = (value - 50) / 1000.0
            current_min, _ = self.touch_processor.get_surface_detection_range()
            if max_offset > current_min:
                self.touch_processor.set_surface_detection_range(
                    current_min, max_offset
                )
            self._update_control_window()

    def _setup_control_window(self) -> None:
        """
        Создание окна управления с ползунками для настройки параметров детекции
        """
        self.control_window_name = "Управление детекцией"
        cv2.namedWindow(self.control_window_name, cv2.WINDOW_AUTOSIZE)

        # Создаем информационное изображение для окна управления
        control_image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(
            control_image,
            "Touch Detection Settings",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
        )

        if self.touch_processor is not None:
            # Ползунок для background_threshold (-1 до 3 метров)
            # # OpenCV ползунки работают только с целыми числами, поэтому используем миллиметры
            # current_bg_threshold = int(
            #     self.touch_processor.get_background_threshold() * 1000
            # )  # Конвертируем в мм
            # cv2.createTrackbar(
            #     "background threshold (mm)",
            #     self.control_window_name,
            #     current_bg_threshold + 1000,  # Смещаем на 1000, чтобы -1000мм стало 0
            #     4000,  # Диапазон: от 0 (=-1000мм) до 4000 (=3000мм)
            #     self._on_background_threshold_change,
            # )

            # # Ползунок для touch_threshold (0 до 0.5 метров = 0 до 500 мм)
            # current_touch_threshold = int(
            #     self.touch_processor.get_touch_threshold() * 1000
            # )
            # cv2.createTrackbar(
            #     "touch threshold (mm)",
            #     self.control_window_name,
            #     current_touch_threshold,
            #     500,  # 0 до 500мм
            #     self._on_touch_threshold_change,
            # )

            # Ползунок включения/выключения разворота на 180 градусов
            flip_enabled = 1 if self.touch_processor.is_flip_180_enabled() else 0
            cv2.createTrackbar(
                "flip 180",
                self.control_window_name,
                flip_enabled,
                1,
                self._on_flip_180_toggle,
            )

            # Ползунок включения/выключения ограничения по поверхности
            surface_filter_enabled = (
                1 if self.touch_processor.is_surface_detection_enabled() else 0
            )
            cv2.createTrackbar(
                "surface filter",
                self.control_window_name,
                surface_filter_enabled,
                1,
                self._on_surface_filter_toggle,
            )

            # Ползунок минимального отклонения (-1000 до +1000 мм → -1.0 до +1.0 м)
            current_min_offset = int(
                self.touch_processor.surface_min_offset * 1000
            )  # в мм
            cv2.createTrackbar(
                "surf min offset (mm)",
                self.control_window_name,
                current_min_offset + 1000,  # смещение: -1000мм → 0
                2000,  # диапазон: -1000..+1000 мм → 0..2000
                self._on_surface_min_offset_change,
            )

            # Ползунок максимального отклонения (-50 до +200 мм → -0.05 до +0.2 м)
            current_max_offset = int(self.touch_processor.surface_max_offset * 1000)
            cv2.createTrackbar(
                "surf max offset (mm)",
                self.control_window_name,
                current_max_offset + 50,  # смещение: -50мм → 0
                500,  # диапазон: -50..+200 мм
                self._on_surface_max_offset_change,
            )

            # Ползунок для смещения глубины (-100 до +200 мм)
            current_depth_offset = int(self.touch_processor.get_depth_offset())
            cv2.createTrackbar(
                "depth offset (mm)",
                self.control_window_name,
                current_depth_offset + 100,  # Смещаем на 100, чтобы -100мм стало 0
                300,  # Диапазон: от 0 (=-100мм) до 300 (=200мм)
                self._on_depth_offset_change,
            )

            # Ползунок для масштабирования глубины (0.8 до 1.5)
            current_scale_factor = int(
                self.touch_processor.get_depth_scale_factor() * 100
            )
            cv2.createTrackbar(
                "depth scale (%)",
                self.control_window_name,
                current_scale_factor,  # 80 до 150 (представляет 0.8 до 1.5)
                150,
                self._on_depth_scale_change,
            )

            # # Ползунок для размера ядра пространственной фильтрации (3 до 15)
            # current_filter_kernel = self.touch_processor.get_spatial_filter_kernel()
            # cv2.createTrackbar(
            #     "filter kernel",
            #     self.control_window_name,
            #     current_filter_kernel,
            #     15,  # 3 до 15
            #     self._on_spatial_filter_kernel_change,
            # )

            # # Ползунок для включения/выключения пространственной фильтрации
            # filter_enabled = (
            #     1 if self.touch_processor.get_spatial_filter_enabled() else 0
            # )
            # cv2.createTrackbar(
            #     "filter on/off",
            #     self.control_window_name,
            #     filter_enabled,
            #     1,  # 0 или 1
            #     self._on_spatial_filter_toggle,
            # )

            # Ползунок для включения/выключения Unity передачи
            unity_enabled = 1 if self.unity_enabled else 0
            cv2.createTrackbar(
                "Unity send",
                self.control_window_name,
                unity_enabled,
                1,  # 0 или 1
                self._on_unity_enabled_toggle,
            )

            # # Ползунок для Unity порта (8000-9000)
            # cv2.createTrackbar(
            #     "Unity port",
            #     self.control_window_name,
            #     self.unity_port - 8000,  # Смещаем базу на 8000
            #     1000,  # 8000 до 9000
            #     self._on_unity_port_change,
            # )

            # Ползунки для TouchFilter
            # Включение/выключение продвинутой фильтрации
            # advanced_filter_enabled = (
            #     1 if self.touch_processor.get_advanced_filter_enabled() else 0
            # )
            # cv2.createTrackbar(
            #     "advanced filter",
            #     self.control_window_name,
            #     advanced_filter_enabled,
            #     1,  # 0 или 1
            #     self._on_advanced_filter_toggle,
            # )

            # # Пороговое расстояние для фильтрации (5-100 пикселей)
            # current_filter_threshold = int(
            #     self.touch_processor.get_filter_distance_threshold()
            # )
            # cv2.createTrackbar(
            #     "filter threshold",
            #     self.control_window_name,
            #     current_filter_threshold - 5,  # Смещаем базу на 5
            #     95,  # 5 до 100
            #     self._on_filter_threshold_change,
            # )

            # # Минимальное движение (1-50 пикселей)
            # current_min_movement = int(self.touch_processor.get_filter_min_movement())
            # cv2.createTrackbar(
            #     "min movement",
            #     self.control_window_name,
            #     current_min_movement - 1,  # Смещаем базу на 1
            #     49,  # 1 до 50
            #     self._on_min_movement_change,
            # )

            # # Таймаут для статичных касаний (0.1-10.0 секунд)
            # current_timeout = int(
            #     self.touch_processor.get_filter_timeout() * 10
            # )  # Конвертируем в десятые доли
            # cv2.createTrackbar(
            #     "touch timeout",
            #     self.control_window_name,
            #     current_timeout - 1,  # Смещаем базу на 1 (0.1с)
            #     99,  # 0.1 до 10.0 секунд
            #     self._on_timeout_change,
            # )

            # # Ползунок для включения/выключения фильтрации по глубине
            # depth_filter_enabled = 1 if self.depth_filter_enabled else 0
            # cv2.createTrackbar(
            #     "depth filter on/off",
            #     self.control_window_name,
            #     depth_filter_enabled,
            #     1,  # 0 или 1
            #     self._on_depth_filter_toggle,
            # )

            # Ползунок для минимальной глубины (10см до 2м = 100мм до 2000мм)
            current_min_depth = int(self.min_touch_depth * 1000)  # Конвертируем в мм
            cv2.createTrackbar(
                "min depth (mm)",
                self.control_window_name,
                current_min_depth - 100,  # Смещаем базу на 100 (100мм->0)
                4000,  # 100мм до 2000мм
                self._on_min_depth_change,
            )

            # Ползунок для максимальной глубины (0.5м до 5м = 500мм до 5000мм)
            current_max_depth = int(self.max_touch_depth * 1000)  # Конвертируем в мм
            cv2.createTrackbar(
                "max depth (mm)",
                self.control_window_name,
                current_max_depth - 500,  # Смещаем базу на 500 (500мм->0)
                4500,  # 500мм до 5000мм
                self._on_max_depth_change,
            )

            # Ползунок для минимальной площади (10–5000 пикселей)
            current_min_area = self.touch_processor.get_min_touch_area()
            cv2.createTrackbar(
                "min touch area",
                self.control_window_name,
                current_min_area,
                5000,
                self._on_min_touch_area_change,
            )

            # Ползунок для максимальной площади (100–20000 пикселей)
            current_max_area = self.touch_processor.get_max_touch_area()
            cv2.createTrackbar(
                "max touch area",
                self.control_window_name,
                current_max_area,
                20000,
                self._on_max_touch_area_change,
            )

            # Добавляем информацию о текущих значениях
            y_pos = 70
            line_height = 25

            # Основные пороги
            cv2.putText(
                control_image,
                f"Background Threshold: {self.touch_processor.get_background_threshold():.3f}m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            y_pos += line_height

            cv2.putText(
                control_image,
                f"Touch Threshold: {self.touch_processor.get_touch_threshold():.3f}m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            y_pos += line_height

            # Коррекция глубины
            cv2.putText(
                control_image,
                f"Depth Offset: {self.touch_processor.get_depth_offset():.1f}mm",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
            )
            y_pos += line_height

            # Ползунок для смещения по X (-50 до +50 пикселей)
            current_offset_x = self.touch_processor.get_touch_offset_x()
            cv2.createTrackbar(
                "touch offset X",
                self.control_window_name,
                current_offset_x + 500,  # Смещаем на 50, чтобы -50 → 0
                1000,  # Диапазон: 0..100 → -50..+50
                self._on_touch_offset_x_change,
            )

            # Ползунок для смещения по Y (-50 до +50 пикселей)
            current_offset_y = self.touch_processor.get_touch_offset_y()
            cv2.createTrackbar(
                "touch offset Y",
                self.control_window_name,
                current_offset_y + 500,
                1000,
                self._on_touch_offset_y_change,
            )

            cv2.putText(
                control_image,
                f"Touch Offset: ({self.touch_processor.get_touch_offset_x()}, {self.touch_processor.get_touch_offset_y()})",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 165, 0),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                f"Depth Scale: {self.touch_processor.get_depth_scale_factor():.2f}x",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
            )
            y_pos += line_height

            # Фильтрация
            filter_status = (
                "ON" if self.touch_processor.get_spatial_filter_enabled() else "OFF"
            )
            cv2.putText(
                control_image,
                f"Spatial Filter: {filter_status} (kernel: {self.touch_processor.get_spatial_filter_kernel()})",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            y_pos += line_height

            # Разделительная линия
            cv2.line(
                control_image, (10, y_pos + 5), (440, y_pos + 5), (100, 100, 100), 1
            )
            y_pos += 20

            # Диапазоны значений
            cv2.putText(
                control_image,
                "Ranges:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            y_pos += line_height

            cv2.putText(
                control_image,
                "Background: -1.0m ... +3.0m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Touch: 0.0m ... 0.5m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Depth Offset: -100mm ... +200mm",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Depth Scale: 0.8x ... 1.5x",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )

            self.logger.info("Окно управления создано")
        else:
            cv2.putText(
                control_image,
                "TouchProcessor не инициализирован",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                control_image,
                "Сначала выполните калибровку",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            self.logger.warning(
                "TouchProcessor не инициализирован - окно управления не создано"
            )

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
            self.logger.debug(
                f"Коэффициент масштабирования изменен: {scale_factor:.2f}"
            )
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

    def _on_flip_180_toggle(self, value: int) -> None:
        if self.touch_processor is not None:
            enabled = bool(value)
            self.touch_processor.enable_flip_180(enabled)
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
        if hasattr(self, "control_window_name") and self.touch_processor is not None:
            # Создаем обновленное информационное изображение
            control_image = np.zeros((550, 500, 3), dtype=np.uint8)
            cv2.putText(
                control_image,
                "Touch Detection Settings",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        y_pos = 350  # Начинаем после существующей информации

        # Разделительная линия
        cv2.line(control_image, (10, y_pos - 10), (440, y_pos - 10), (100, 100, 100), 1)
        y_pos += 10

        # Заголовок
        cv2.putText(
            control_image,
            "Depth Filtering:",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        y_pos += 25

        cv2.putText(
            control_image,
            f"Flip 180: {'ON' if self.touch_processor.is_flip_180_enabled() else 'OFF'}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 165, 0),
            1,
        )
        y_pos += 20

        cv2.putText(
            control_image,
            f"Surface Filter: {'ON' if self.touch_processor.is_surface_detection_enabled() else 'OFF'}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 165, 0),
            1,
        )
        y_pos += 20

        min_off, max_off = self.touch_processor.get_surface_detection_range()
        cv2.putText(
            control_image,
            f"Surface Range: {min_off*1000:.0f} to {max_off*1000:.0f} mm",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 165, 0),
            1,
        )
        y_pos += 20

        # Статус фильтрации
        filter_status = "ON" if self.depth_filter_enabled else "OFF"
        status_color = (0, 255, 0) if self.depth_filter_enabled else (0, 0, 255)
        cv2.putText(
            control_image,
            f"Depth Filter: {filter_status}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
        )
        y_pos += 20

        # Диапазон глубины
        cv2.putText(
            control_image,
            f"Depth Range: {self.min_touch_depth:.2f}-{self.max_touch_depth:.2f}m",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 165, 0),
            1,
        )
        y_pos += 20

        # Статистика глубины (если доступна)
        if (
            hasattr(self, "_last_cropped_depth")
            and self._last_cropped_depth is not None
        ):
            depth_stats = self.touch_processor.get_depth_filter_stats(
                self._last_cropped_depth
            )
            if depth_stats and depth_stats["valid_pixels"] > 0:
                cv2.putText(
                    control_image,
                    f"Current Depth: {depth_stats['mean_depth']:.2f}m",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )
                y_pos += 15

                cv2.putText(
                    control_image,
                    f"Min/Max: {depth_stats['min_depth']:.2f}m / {depth_stats['max_depth']:.2f}m",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )
                y_pos += 15

            # Добавляем информацию о текущих значениях
            y_pos = 70
            line_height = 25

            # Основные пороги
            cv2.putText(
                control_image,
                f"Background Threshold: {self.touch_processor.get_background_threshold():.3f}m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            y_pos += line_height

            cv2.putText(
                control_image,
                f"Touch Threshold: {self.touch_processor.get_touch_threshold():.3f}m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            y_pos += line_height

            # Коррекция глубины
            cv2.putText(
                control_image,
                f"Depth Offset: {self.touch_processor.get_depth_offset():.1f}mm",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
            )
            y_pos += line_height

            cv2.putText(
                control_image,
                f"Depth Scale: {self.touch_processor.get_depth_scale_factor():.2f}x",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
            )
            y_pos += line_height

            # Фильтрация
            filter_status = (
                "ON" if self.touch_processor.get_spatial_filter_enabled() else "OFF"
            )
            cv2.putText(
                control_image,
                f"Spatial Filter: {filter_status} (kernel: {self.touch_processor.get_spatial_filter_kernel()})",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            y_pos += line_height

            # Разделительная линия
            cv2.line(
                control_image, (10, y_pos + 5), (440, y_pos + 5), (100, 100, 100), 1
            )
            y_pos += 20

            # Диапазоны значений
            cv2.putText(
                control_image,
                "Ranges:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            y_pos += line_height

            cv2.putText(
                control_image,
                "Background: -1.0m ... +3.0m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Touch: 0.0m ... 0.5m",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Depth Offset: -100mm ... +200mm",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Depth Scale: 0.8x ... 1.5x",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y_pos += 20

            # TouchFilter информация
            filter_status = (
                "ON" if self.touch_processor.get_advanced_filter_enabled() else "OFF"
            )
            cv2.putText(
                control_image,
                f"Advanced Filter: {filter_status}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 165, 0),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                f"Filter Threshold: {self.touch_processor.get_filter_distance_threshold():.0f}px",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 165, 0),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                f"Min Movement: {self.touch_processor.get_filter_min_movement():.0f}px",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 165, 0),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                f"Touch Timeout: {self.touch_processor.get_filter_timeout():.1f}s",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 165, 0),
                1,
            )
            y_pos += 20

            # Статистика фильтра
            filter_stats = self.touch_processor.get_filter_statistics()
            cv2.putText(
                control_image,
                f"Filtered: {filter_stats['total_filtered']}/{filter_stats['total_processed']}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                f"Active Touches: {filter_stats['active_touches']}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
            y_pos += 20

            cv2.putText(
                control_image,
                "Press 'ESC' to exit",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                control_image,
                f"Touch Area: {self.touch_processor.get_min_touch_area()}–{self.touch_processor.get_max_touch_area()} px",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            y_pos += line_height

            # Отображаем обновленное изображение
            cv2.imshow(self.control_window_name, control_image)

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """
        Обработчик событий мыши для калибровки поверхности

        Args:
            event: Тип события мыши
            x: X координата курсора
            y: Y координата курсора
            flags: Дополнительные флаги
            param: Дополнительные параметры
        """
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and self.surface_calibration_mode
            and self.touch_processor is not None
            and self.show_cropped
        ):

            # Проверяем, что клик в области обрезанного изображения (левая часть)
            if x < self.crop_width:  # Ширина обрезанного изображения
                # Получаем текущие обрезанные изображения
                if (
                    hasattr(self, "_last_cropped_depth")
                    and self._last_cropped_depth is not None
                ):
                    success = self.touch_processor.add_surface_calibration_point(
                        x, y, self._last_cropped_depth
                    )
                    if success:
                        points_collected = len(
                            self.touch_processor.surface_calibration_points
                        )
                        self.logger.info(
                            f"Точек калибровки собрано: {points_collected}/{self.surface_calibration_points_needed}"
                        )

                        # Автоматическая калибровка после сбора достаточного количества точек
                        if points_collected >= self.surface_calibration_points_needed:
                            # ЗАМЕНИТЕ ЭТУ СТРОКУ:
                            # if self.touch_processor.calibrate_surface_height():
                            # НА ЭТУ:
                            if (
                                self.touch_processor.calibrate_surface_height()
                            ):  # Метод уже обновлен!
                                self.surface_calibration_mode = False

                                # Получаем информацию о качестве калибровки
                                quality = (
                                    self.touch_processor.get_plane_calibration_quality()
                                )
                                self.logger.info(
                                    f"Калибровка поверхности завершена! Качество: {quality['quality']}"
                                )
                                self.logger.info(
                                    f"Уравнение плоскости: {quality['equation']}"
                                )
                                self.logger.info(
                                    f"Точность: RMSE={quality['rmse']:.4f}м"
                                )

                                # Предупреждение если точность низкая
                                if quality["quality"] in ["fair", "poor"]:
                                    self.logger.warning(
                                        "Низкая точность калибровки! Рекомендуется перекалибровать поверхность."
                                    )
                            else:
                                self.logger.warning(
                                    "Не удалось завершить калибровку поверхности"
                                )
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
            self.touch_processor.set_depth_filter_range(
                self.min_touch_depth, self.max_touch_depth
            )

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
                self.touch_processor.set_depth_filter_range(
                    self.min_touch_depth, self.max_touch_depth
                )

            self._update_control_window()
        else:
            self.logger.warning(
                "Минимальная глубина должна быть меньше максимальной хотя бы на 10см"
            )

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
                self.touch_processor.set_depth_filter_range(
                    self.min_touch_depth, self.max_touch_depth
                )

            self._update_control_window()
        else:
            self.logger.warning(
                "Максимальная глубина должна быть больше минимальной хотя бы на 10см"
            )

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

    def _create_display_image(
        self, color_image: np.ndarray, depth_image: np.ndarray
    ) -> np.ndarray:
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
            cropped_color, cropped_depth = self.cropper.crop_both_images(
                color_image, depth_image
            )
            # Применяем разворот на 180 градусов, если включён
            if self.touch_processor and self.touch_processor.is_flip_180_enabled():
                cropped_color = cv2.rotate(cropped_color, cv2.ROTATE_180)
                cropped_depth = cv2.rotate(cropped_depth, cv2.ROTATE_180)

            if cropped_color is not None and cropped_depth is not None:
                # Применяем цветовую карту к обрезанному изображению глубины
                depth_colormap: np.ndarray = self.camera.apply_colormap_to_depth(
                    cropped_depth
                )

                # Создаем комбинированное изображение
                if self.show_depth:
                    images: np.ndarray = np.hstack((cropped_color, depth_colormap))
                else:
                    images: np.ndarray = np.hstack((cropped_color))

                # Добавляем текст с информацией
                cv2.putText(
                    images,
                    "Cropped Color",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                if self.show_depth:
                    cv2.putText(
                        images,
                        "Cropped Depth",
                        (cropped_color.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Добавляем информацию о размере
                cv2.putText(
                    images,
                    f"Size: {cropped_color.shape[1]}x{cropped_color.shape[0]}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                # Информация о фильтрации по глубине
                depth_filter_status = "ON" if self.depth_filter_enabled else "OFF"
                depth_filter_color = (
                    (0, 255, 0) if self.depth_filter_enabled else (0, 0, 255)
                )
                cv2.putText(
                    images,
                    f"Depth Filter: {depth_filter_status} ({self.min_touch_depth:.1f}-{self.max_touch_depth:.1f}m)",
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    depth_filter_color,
                    1,
                )

                # Сохраняем последние обрезанные изображения для калибровки поверхности
                self._last_cropped_depth = cropped_depth

                # Обрабатываем касания если включена детекция
                if self.touch_detection_enabled and self.touch_processor is not None:
                    self.current_touches = self.touch_processor.process_frame(
                        cropped_color, cropped_depth
                    )

                    # Отправляем касания в Unity если включена передача
                    if (
                        self.unity_enabled
                        and self.unity_comm is not None
                        and self.current_touches
                    ):
                        self.unity_comm.send_touch_coordinates(self.current_touches)

                    # Добавляем отладочные изображения если включен режим отладки
                    if self.show_debug_images:
                        debug_images = self.touch_processor.get_debug_images()
                        debug_list = []

                        # Добавляем бинарное изображение зоны детекции
                        if debug_images["binary"] is not None:
                            binary_colored = cv2.applyColorMap(
                                debug_images["binary"], cv2.COLORMAP_HOT
                            )
                            # Добавляем подпись
                            cv2.putText(
                                binary_colored,
                                "Binary Detection",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                            debug_list.append(binary_colored)

                        # Добавляем нормализованное разностное изображение
                        if debug_images["depth_diff_normalized"] is not None:
                            diff_colored = cv2.applyColorMap(
                                debug_images["depth_diff_normalized"],
                                cv2.COLORMAP_VIRIDIS,
                            )
                            # Добавляем подпись
                            cv2.putText(
                                diff_colored,
                                "Depth Difference",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                            debug_list.append(diff_colored)

                        # Объединяем с основными изображениями
                        if debug_list:
                            images = np.hstack([images] + debug_list)

                    # Визуализируем касания на обрезанном изображении
                    if self.show_touches and self.current_touches:
                        images = self._visualize_touches_on_cropped(
                            images, cropped_color
                        )

                # Визуализация режима калибровки поверхности
                if self.surface_calibration_mode and self.touch_processor is not None:
                    images = self._visualize_surface_calibration(images)

                # Добавляем информацию о касаниях и режиме отладки
                touch_status = "ON" if self.touch_detection_enabled else "OFF"
                cv2.putText(
                    images,
                    f"Touch Detection: {touch_status}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if self.touch_detection_enabled else (0, 0, 255),
                    1,
                )

                # Информация о режиме отладки
                debug_status = "ON" if self.show_debug_images else "OFF"
                cv2.putText(
                    images,
                    f"Debug Mode: {debug_status}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0) if self.show_debug_images else (128, 128, 128),
                    1,
                )

                if self.current_touches:
                    cv2.putText(
                        images,
                        f"Touches: {len(self.current_touches)}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                # Добавляем информацию о чувствительности
                if self.touch_processor is not None:
                    sensitivity_text = (
                        f"Sensitivity: {self.touch_processor.sensitivity_level}/10"
                    )
                    cv2.putText(
                        images,
                        sensitivity_text,
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                    # Информация о калибровке поверхности
                    surface_status = (
                        "Calibrated"
                        if self.touch_processor.surface_height is not None
                        else "Not Calibrated"
                    )
                    surface_color = (
                        (0, 255, 0)
                        if self.touch_processor.surface_height is not None
                        else (0, 0, 255)
                    )
                    cv2.putText(
                        images,
                        f"Surface: {surface_status}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        surface_color,
                        1,
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
                "Q - exit program",
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
                "Q - exit program",
                "I - flip image 180°",
            ]

        y_offset = images.shape[0] // 2 - 100
        for instruction in instructions:
            cv2.putText(
                images,
                instruction,
                (8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Черная обводка
                3,  # Толщина обводки
                cv2.LINE_AA,
            )
            cv2.putText(
                images,
                instruction,
                (8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            y_offset += 20

        return images

    def _create_original_display_image(
        self, color_image: np.ndarray, depth_image: np.ndarray
    ) -> np.ndarray:
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
            "Original Color",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            images,
            "Original Depth",
            (color_image.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Добавляем статус калибровки
        calibration_status = (
            "Calibrated" if self.cropper.is_calibrated else "Not Calibrated"
        )
        cv2.putText(
            images,
            f"Status: {calibration_status}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if self.cropper.is_calibrated else (0, 0, 255),
            1,
        )

        return images

    def _visualize_touches_on_cropped(
        self, combined_image: np.ndarray, cropped_color: np.ndarray
    ) -> np.ndarray:
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
                2,
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
                1,
            )

            # Показываем также на depth части (правая часть изображения)
            depth_x = orig_x + cropped_color.shape[1]
            cv2.circle(result, (depth_x, orig_y), 8, color, 2)

            # Выводим информацию о касании в консоль
            self.logger.info(
                f"Touch {i+1}: screen({scaled_x},{scaled_y}) crop({orig_x},{orig_y}) conf={confidence:.2f}"
            )

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
            for i, (x, y, depth) in enumerate(
                self.touch_processor.surface_calibration_points
            ):
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
                    2,
                )

            # Добавляем информацию о режиме калибровки
            cv2.putText(
                result,
                "SURFACE CALIBRATION MODE",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

            points_text = f"Points: {len(self.touch_processor.surface_calibration_points)}/{self.surface_calibration_points_needed}"
            cv2.putText(
                result,
                points_text,
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

        return result

    def _on_min_touch_area_change(self, value: int) -> None:
        if self.touch_processor is not None:
            self.touch_processor.set_min_touch_area(value)
            # Обновляем max, чтобы он не был меньше min
            current_max = self.touch_processor.get_max_touch_area()
            if current_max < value:
                self.touch_processor.set_max_touch_area(value)
                cv2.setTrackbarPos("max touch area", self.control_window_name, value)
            self._update_control_window()

    def _on_max_touch_area_change(self, value: int) -> None:
        if self.touch_processor is not None:
            self.touch_processor.set_max_touch_area(value)
            # Обновляем min, чтобы он не превышал max
            current_min = self.touch_processor.get_min_touch_area()
            if current_min > value:
                self.touch_processor.set_min_touch_area(value)
                cv2.setTrackbarPos("min touch area", self.control_window_name, value)
            self._update_control_window()

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
            self.cropper = ProjectionAreaCropper(
                output_size=(self.crop_width, self.crop_height)
            )
            if self.cropper.is_calibrated:
                self.logger.info("Обрезчик проекции обновлен с новой калибровкой")
                # Переинициализируем процессор касаний
                self._init_touch_processor()
            else:
                self.logger.warning(
                    "Не удалось загрузить новые данные калибровки в обрезчик"
                )
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
                    self.logger.info(
                        f"Внутренние параметры цветной камеры: {color_intrinsics.width}x{color_intrinsics.height}"
                    )
                    self.logger.info(
                        f"Внутренние параметры камеры глубины: {depth_intrinsics.width}x{depth_intrinsics.height}"
                    )

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
            if key == ord("q"):
                break
                # НОВЫЕ ОБРАБОТЧИКИ ДЛЯ ФИЛЬТРАЦИИ ПО ГЛУБИНЕ
            elif key == ord("f"):  # Включение/выключение фильтрации по глубине
                self.depth_filter_enabled = not self.depth_filter_enabled
                status = "включена" if self.depth_filter_enabled else "выключена"
                self.logger.info(f"Фильтрация по глубине {status}")

                if self.touch_processor is not None:
                    self.touch_processor.enable_depth_filter(self.depth_filter_enabled)

            elif key == ord("1"):  # Увеличить минимальную глубину
                if self.touch_processor is not None:
                    new_min = min(
                        self.max_touch_depth - 0.1, self.min_touch_depth + 0.1
                    )
                    if new_min != self.min_touch_depth:
                        self.min_touch_depth = new_min
                        self.touch_processor.set_depth_filter_range(
                            self.min_touch_depth, self.max_touch_depth
                        )
                        self.logger.info(
                            f"Минимальная глубина увеличена до: {self.min_touch_depth:.2f}м"
                        )

            elif key == ord("i"):
                if self.touch_processor is not None:
                    new_state = not self.touch_processor.is_flip_180_enabled()
                    self.touch_processor.enable_flip_180(new_state)
                    status = "включён" if new_state else "выключён"
                    self.logger.info(f"Разворот изображения на 180° {status}")

            elif key == ord("2"):  # Уменьшить минимальную глубину
                if self.touch_processor is not None:
                    new_min = max(0.1, self.min_touch_depth - 0.1)
                    if new_min != self.min_touch_depth:
                        self.min_touch_depth = new_min
                        self.touch_processor.set_depth_filter_range(
                            self.min_touch_depth, self.max_touch_depth
                        )
                        self.logger.info(
                            f"Минимальная глубина уменьшена до: {self.min_touch_depth:.2f}м"
                        )

            elif key == ord("3"):  # Увеличить максимальную глубину
                if self.touch_processor is not None:
                    new_max = min(5.0, self.max_touch_depth + 0.1)
                    if new_max != self.max_touch_depth:
                        self.max_touch_depth = new_max
                        self.touch_processor.set_depth_filter_range(
                            self.min_touch_depth, self.max_touch_depth
                        )
                        self.logger.info(
                            f"Максимальная глубина увеличена до: {self.max_touch_depth:.2f}м"
                        )

            elif key == ord("4"):  # Уменьшить максимальную глубину
                if self.touch_processor is not None:
                    new_max = max(
                        self.min_touch_depth + 0.1, self.max_touch_depth - 0.1
                    )
                    if new_max != self.max_touch_depth:
                        self.max_touch_depth = new_max
                        self.touch_processor.set_depth_filter_range(
                            self.min_touch_depth, self.max_touch_depth
                        )
                        self.logger.info(
                            f"Максимальная глубина уменьшена до: {self.max_touch_depth:.2f}м"
                        )

            elif key == ord("5"):  # Автоматическая настройка диапазона глубины
                if (
                    self.touch_processor is not None
                    and hasattr(self.touch_processor, "plane_A")
                    and self.touch_processor.plane_A is not None
                ):

                    # Автоматическая настройка на основе калиброванной поверхности
                    if (
                        hasattr(self, "_last_cropped_depth")
                        and self._last_cropped_depth is not None
                    ):
                        self.touch_processor.auto_set_depth_range(
                            self._last_cropped_depth, margin=0.2
                        )

                        # Обновляем наши переменные
                        self.min_touch_depth = self.touch_processor.min_touch_depth
                        self.max_touch_depth = self.touch_processor.max_touch_depth
                        self.logger.info(
                            f"Автоматически установлен диапазон глубины: {self.min_touch_depth:.2f}-{self.max_touch_depth:.2f}м"
                        )
                else:
                    self.logger.warning(
                        "Для автоматической настройки нужна калиброванная поверхность"
                    )
            elif key == ord("c"):
                # Переход в режим калибровки
                self._enter_calibration_mode()
            elif key == ord("v"):
                # Переключение между исходным и обрезанным видом
                if self.cropper.is_calibrated:
                    self.show_cropped = not self.show_cropped
                    mode = "cropped" if self.show_cropped else "original"
                    self.logger.info(f"Переключен режим отображения: {mode}")
                else:
                    self.logger.warning(
                        "Сначала выполните калибровку для работы с обрезанными изображениями"
                    )
            elif key == ord("p"):
                # Переключение отображения области проекции
                self.show_projection_area = not self.show_projection_area
                status = "включено" if self.show_projection_area else "выключено"
                self.logger.info(f"Отображение области проекции: {status}")
            elif key == ord("t"):
                # Переключение детекции касаний
                if self.touch_processor is not None:
                    self.touch_detection_enabled = not self.touch_detection_enabled
                    status = "включена" if self.touch_detection_enabled else "выключена"
                    self.logger.info(f"Детекция касаний: {status}")
                    if not self.touch_detection_enabled:
                        self.current_touches.clear()
                else:
                    self.logger.warning(
                        "TouchProcessor не инициализирован. Сначала выполните калибровку."
                    )
            elif key == ord("b"):
                # Установка фонового изображения для детекции касаний
                if (
                    self.touch_detection_enabled
                    and self.touch_processor is not None
                    and self.show_cropped
                ):
                    # Получаем текущие обрезанные изображения
                    cropped_color, cropped_depth = self.cropper.crop_both_images(
                        color_image, depth_image
                    )
                    if cropped_depth is not None:
                        self.touch_processor.set_background(cropped_depth)
                        self.logger.info(
                            "Фоновое изображение для детекции касаний установлено"
                        )
                    else:
                        self.logger.warning(
                            "Не удалось получить обрезанное изображение глубины"
                        )
                else:
                    self.logger.warning(
                        "Для установки фона включите детекцию касаний и режим обрезанного вида"
                    )
            elif key == ord("r"):
                # Сброс фонового изображения
                if self.touch_processor is not None:
                    self.touch_processor.reset_background()
                    self.current_touches.clear()
                    self.logger.info("Фоновое изображение сброшено")
                else:
                    self.logger.warning("TouchProcessor не инициализирован")
            elif key == ord("d"):
                # Переключение режима отладки
                self.show_debug_images = not self.show_debug_images
                status = "ВКЛЮЧЕН" if self.show_debug_images else "ВЫКЛЮЧЕН"
                self.logger.info(f"Режим отладки {status}")

                if self.show_debug_images:
                    self.logger.info(
                        "Показывать отладочные изображения: бинарная маска детекции и разность глубины"
                    )
                    if not self.touch_detection_enabled:
                        self.logger.warning(
                            "Для отображения отладочных изображений включите детекцию касаний (T)"
                        )
                    if not self.show_cropped:
                        self.logger.warning(
                            "Для отображения отладочных изображений переключитесь в режим кропнутого вида (V)"
                        )
            elif key == ord("s"):
                # Вход в режим калибровки поверхности
                if self.touch_processor is not None and self.show_cropped:
                    if not self.surface_calibration_mode:
                        self.surface_calibration_mode = True
                        self.touch_processor.clear_surface_calibration()
                        self.logger.info(
                            "Начата калибровка поверхности. Кликните на 5 точек поверхности проекции."
                        )
                    else:
                        # Завершаем калибровку вручную
                        if len(self.touch_processor.surface_calibration_points) >= 3:
                            if self.touch_processor.calibrate_surface_height():
                                self.surface_calibration_mode = False
                                self.logger.info(
                                    "Калибровка поверхности завершена вручную"
                                )
                            else:
                                self.logger.warning(
                                    "Не удалось завершить калибровку поверхности"
                                )
                        else:
                            self.logger.warning(
                                "Недостаточно точек для калибровки (минимум 3)"
                            )
                else:
                    self.logger.warning(
                        "Для калибровки поверхности включите обрезанный вид и инициализируйте TouchProcessor"
                    )
            elif key == ord("+") or key == ord("="):
                # Увеличение чувствительности
                if self.touch_processor is not None:
                    self.touch_processor.adjust_sensitivity(1)
                else:
                    self.logger.warning("TouchProcessor не инициализирован")
            elif key == ord("-"):
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
    camera: RealSenseCamera = RealSenseCamera(width=1280, height=720, fps=15)

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
