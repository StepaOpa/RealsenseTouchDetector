"""
Модуль для работы с камерой Intel RealSense D435F
Предоставляет функциональность для получения цветных изображений и изображений глубины
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Tuple, Optional, Any
import logging


class RealSenseCamera:
    """
    Класс для работы с камерой Intel RealSense D435F
    Позволяет получать цветные изображения и изображения глубины
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        """
        Инициализация камеры RealSense
        
        Args:
            width: Ширина изображения
            height: Высота изображения  
            fps: Частота кадров
        """
        self.width: int = width
        self.height: int = height
        self.fps: int = fps
        
        # Инициализируем pipeline и config
        self.pipeline: rs.pipeline = rs.pipeline()
        self.config: rs.config = rs.config()
        
        # Настраиваем потоки для color и depth
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # Align объект для выравнивания изображений глубины с цветными
        self.align: rs.align = rs.align(rs.stream.color)
        
        # Флаг состояния камеры
        self.is_running: bool = False
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger: logging.Logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """
        Запуск камеры
        
        Returns:
            True если камера успешно запущена, False в противном случае
        """
        try:
            # Запускаем pipeline
            self.pipeline.start(self.config)
            self.is_running = True
            self.logger.info("Камера RealSense D435F успешно запущена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка запуска камеры: {e}")
            return False
    
    def stop(self) -> None:
        """
        Остановка камеры
        """
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            self.logger.info("Камера RealSense D435F остановлена")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Получение кадров с камеры
        
        Returns:
            Кортеж (color_image, depth_image) где:
            - color_image: цветное изображение в формате BGR
            - depth_image: изображение глубины в формате uint16
            Возвращает (None, None) в случае ошибки
        """
        if not self.is_running:
            self.logger.warning("Камера не запущена")
            return None, None
        
        try:
            # Получаем frames
            frames: rs.composite_frame = self.pipeline.wait_for_frames()
            
            # Выравниваем depth frame к color frame
            aligned_frames: rs.composite_frame = self.align.process(frames)
            
            # Получаем выровненные кадры
            depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
            color_frame: rs.video_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Конвертируем в numpy arrays
            depth_image: np.ndarray = np.asanyarray(depth_frame.get_data())
            color_image: np.ndarray = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            self.logger.error(f"Ошибка получения кадров: {e}")
            return None, None
    
    def get_depth_scale(self) -> float:
        """
        Получение масштаба глубины камеры
        
        Returns:
            Масштаб глубины (обычно 0.001 для миллиметров)
        """
        if not self.is_running:
            return 0.0
        
        try:
            profile: rs.pipeline_profile = self.pipeline.get_active_profile()
            depth_sensor: rs.depth_sensor = profile.get_device().first_depth_sensor()
            return depth_sensor.get_depth_scale()
        except Exception as e:
            self.logger.error(f"Ошибка получения масштаба глубины: {e}")
            return 0.0
    
    def get_intrinsics(self) -> Tuple[Optional[rs.intrinsics], Optional[rs.intrinsics]]:
        """
        Получение внутренних параметров камеры
        
        Returns:
            Кортеж (color_intrinsics, depth_intrinsics)
        """
        if not self.is_running:
            return None, None
        
        try:
            profile: rs.pipeline_profile = self.pipeline.get_active_profile()
            
            color_stream: rs.video_stream_profile = profile.get_stream(rs.stream.color)
            depth_stream: rs.video_stream_profile = profile.get_stream(rs.stream.depth)
            
            color_intrinsics: rs.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            depth_intrinsics: rs.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            return color_intrinsics, depth_intrinsics
            
        except Exception as e:
            self.logger.error(f"Ошибка получения внутренних параметров: {e}")
            return None, None
    
    def apply_colormap_to_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Применение цветовой карты к изображению глубины для визуализации
        
        Args:
            depth_image: Изображение глубины
            
        Returns:
            Цветное изображение глубины
        """
        # Нормализуем изображение глубины для лучшей визуализации
        depth_colormap: np.ndarray = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        return depth_colormap
    
    def __enter__(self) -> 'RealSenseCamera':
        """
        Контекстный менеджер для автоматического управления ресурсами
        """
        if self.start():
            return self
        else:
            raise RuntimeError("Не удалось запустить камеру RealSense")
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Закрытие камеры при выходе из контекста
        """
        self.stop()
