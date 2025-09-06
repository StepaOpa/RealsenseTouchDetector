"""
Модуль для управления камерой Intel RealSense D435f
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import time


class RealSenseManager:
    """Класс для управления камерой Intel RealSense"""
    
    def __init__(self, width: int = 640, height: int =480, fps: int = 30) -> None:
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
        
        # Инициализация pipeline и config
        self.pipeline: rs.pipeline = rs.pipeline()
        self.config: rs.config = rs.config()
        
        # Настройка потоков
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Фильтры для улучшения качества depth изображения
        self.align = rs.align(rs.stream.color)
        self.depth_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_fill_filter = rs.hole_filling_filter()
        
        self.profile: Optional[rs.pipeline_profile] = None
        self.depth_scale: float = 0.0
        self.intrinsics: Optional[rs.intrinsics] = None
        
    def start(self) -> bool:
        """
        Запуск камеры
        
        Returns:
            True если камера успешно запущена, False в противном случае
        """
        try:
            # Проверка наличия подключенных устройств
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("Ошибка: RealSense устройство не найдено!")
                return False
                
            # Запуск pipeline
            self.profile = self.pipeline.start(self.config)
            
            # Получение масштаба глубины
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Получение intrinsics для калибровки
            depth_stream = self.profile.get_stream(rs.stream.depth)
            self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print(f"Камера RealSense запущена успешно!")
            print(f"Разрешение: {self.width}x{self.height} @ {self.fps} FPS")
            print(f"Масштаб глубины: {self.depth_scale}")
            
            # Пропуск первых кадров для стабилизации
            for _ in range(30):
                self.pipeline.wait_for_frames()
                
            return True
            
        except Exception as e:
            print(f"Ошибка запуска камеры: {e}")
            return False
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Получение кадров глубины и цвета
        
        Returns:
            Tuple с depth_image и color_image (или None при ошибке)
        """
        try:
            # Ожидание кадров
            frames = self.pipeline.wait_for_frames()
            
            # Выравнивание depth по color
            aligned_frames = self.align.process(frames)
            
            # Получение кадров
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Применение фильтров к depth
            depth_frame = self.depth_filter.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_fill_filter.process(depth_frame)
            
            # Конвертация в numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return depth_image, color_image
            
        except Exception as e:
            print(f"Ошибка получения кадров: {e}")
            return None, None
    
    def get_depth_at_pixel(self, x: int, y: int, depth_image: np.ndarray) -> float:
        """
        Получение глубины в конкретном пикселе
        
        Args:
            x: Координата X
            y: Координата Y
            depth_image: Изображение глубины
            
        Returns:
            Глубина в метрах
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            depth_value = depth_image[y, x]
            return depth_value * self.depth_scale
        return 0.0
    
    def pixel_to_3d(self, x: int, y: int, depth: float) -> Tuple[float, float, float]:
        """
        Преобразование пиксельных координат в 3D координаты
        
        Args:
            x: Пиксельная координата X
            y: Пиксельная координата Y
            depth: Глубина в метрах
            
        Returns:
            3D координаты (X, Y, Z) в метрах
        """
        if self.intrinsics is None:
            return 0.0, 0.0, 0.0
            
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return float(point_3d[0]), float(point_3d[1]), float(point_3d[2])
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Получение информации о камере
        
        Returns:
            Словарь с параметрами камеры
        """
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'depth_scale': self.depth_scale,
            'intrinsics': self.intrinsics.__dict__ if self.intrinsics else None
        }
    
    def stop(self) -> None:
        """Остановка камеры"""
        try:
            if self.profile is not None:
                self.pipeline.stop()
                self.profile = None
                print("Камера RealSense остановлена")
        except Exception as e:
            print(f"Ошибка остановки камеры: {e}")
    
    def __del__(self) -> None:
        """Деструктор"""
        self.stop()


# Функция для тестирования камеры
def test_camera() -> None:
    """Тестирование работы камеры"""
    camera = RealSenseManager()
    
    if not camera.start():
        return
    
    try:
        print("Нажмите ESC для выхода")
        while True:
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                continue
            
            # Создание цветной карты глубины для визуализации
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Приведение изображений к одному размеру для корректного склеивания
            color_height, color_width = color_image.shape[:2]
            depth_height, depth_width = depth_colormap.shape[:2]
            
            # Если размеры не совпадают, изменяем размер карты глубины
            if (color_height != depth_height) or (color_width != depth_width):
                depth_colormap = cv2.resize(depth_colormap, (color_width, color_height))
                print(f"Изменен размер карты глубины: {depth_width}x{depth_height} -> {color_width}x{color_height}")
            
            # Объединение изображений
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense Test - Color | Depth', images)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()
