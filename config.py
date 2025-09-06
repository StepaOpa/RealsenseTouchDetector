"""
Конфигурационный файл для системы отслеживания ног
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import json
import os


@dataclass
class CameraConfig:
    """Конфигурация камеры"""
    width: int = 640
    height: int = 480
    fps: int = 30
    # Настройки фильтров глубины
    enable_decimation: bool = True
    enable_spatial: bool = True
    enable_temporal: bool = True
    enable_hole_filling: bool = True


@dataclass
class FloorDetectionConfig:
    """Конфигурация детекции пола"""
    floor_depth: float = 1.5  # метры
    tolerance: float = 0.2    # метры
    object_height_offset: float = 0.05  # на сколько объекты выше пола
    
    # Морфологические операции
    morphology_kernel_size: int = 5
    close_kernel_size: int = 15
    
    # Фильтрация контуров
    min_contour_area: int = 500
    max_contour_area: int = 50000
    max_compactness: float = 50.0  # Фильтр по форме контура
    
    # Медианный фильтр
    median_blur_kernel: int = 5


@dataclass
class CalibrationConfig:
    """Конфигурация калибровки"""
    target_width: int = 1920
    target_height: int = 1080
    calibration_file: str = "calibration.json"
    auto_load_calibration: bool = True
    auto_save_on_complete: bool = True


@dataclass
class TrackingConfig:
    """Конфигурация трекинга"""
    max_feet_count: int = 10
    min_confidence: float = 0.3
    tracking_history_size: int = 10
    
    # Параметры для вычисления уверенности
    confidence_area_min: float = 500.0
    confidence_area_max: float = 20000.0
    confidence_compactness_threshold: float = 50.0
    confidence_depth_max: float = 5.0


@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""
    show_trails: bool = True
    trail_length: int = 30
    trail_max_age: float = 5.0  # секунды
    
    show_grid: bool = True
    grid_step: int = 100
    grid_major_step: int = 500
    
    show_zones: bool = False
    show_info_panel: bool = True
    
    # Цвета для стоп (BGR)
    foot_colors: List[Tuple[int, int, int]] = None
    
    # Размеры элементов
    camera_foot_radius_base: int = 8
    camera_foot_radius_confidence_scale: int = 10
    screen_foot_radius_base: int = 30
    
    def __post_init__(self):
        if self.foot_colors is None:
            self.foot_colors = [
                (100, 100, 255),  # Красный (BGR)
                (100, 255, 100),  # Зеленый
                (255, 100, 100),  # Синий
                (100, 255, 255),  # Желтый
                (255, 100, 255),  # Магента
                (255, 255, 100),  # Циан
                (50, 150, 255),   # Оранжевый
                (255, 50, 150),   # Фиолетовый
            ]


@dataclass
class SystemConfig:
    """Общая конфигурация системы"""
    # Подкомпоненты
    camera: CameraConfig
    floor_detection: FloorDetectionConfig
    calibration: CalibrationConfig
    tracking: TrackingConfig
    visualization: VisualizationConfig
    
    # Общие настройки
    debug_mode: bool = True
    verbose_logging: bool = True
    
    # Файлы
    config_file: str = "system_config.json"
    log_file: str = "foot_tracking.log"
    
    # Производительность
    target_fps: float = 30.0
    performance_monitoring: bool = True


class ConfigManager:
    """Менеджер конфигурации"""
    
    def __init__(self, config_file: str = "system_config.json"):
        self.config_file = config_file
        self.config = self.load_default_config()
    
    def load_default_config(self) -> SystemConfig:
        """Загрузка конфигурации по умолчанию"""
        return SystemConfig(
            camera=CameraConfig(),
            floor_detection=FloorDetectionConfig(),
            calibration=CalibrationConfig(),
            tracking=TrackingConfig(),
            visualization=VisualizationConfig()
        )
    
    def save_config(self, filename: str = None) -> bool:
        """
        Сохранение конфигурации в файл
        
        Args:
            filename: Имя файла для сохранения
            
        Returns:
            True если сохранение прошло успешно
        """
        if filename is None:
            filename = self.config_file
            
        try:
            # Преобразование конфигурации в словарь
            config_dict = self._config_to_dict(self.config)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"Конфигурация сохранена в файл: {filename}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False
    
    def load_config(self, filename: str = None) -> bool:
        """
        Загрузка конфигурации из файла
        
        Args:
            filename: Имя файла для загрузки
            
        Returns:
            True если загрузка прошла успешно
        """
        if filename is None:
            filename = self.config_file
            
        if not os.path.exists(filename):
            print(f"Файл конфигурации не найден: {filename}")
            return False
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Обновление конфигурации из словаря
            self.config = self._dict_to_config(config_dict)
            
            print(f"Конфигурация загружена из файла: {filename}")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return False
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        return {
            'camera': {
                'width': config.camera.width,
                'height': config.camera.height,
                'fps': config.camera.fps,
                'enable_decimation': config.camera.enable_decimation,
                'enable_spatial': config.camera.enable_spatial,
                'enable_temporal': config.camera.enable_temporal,
                'enable_hole_filling': config.camera.enable_hole_filling,
            },
            'floor_detection': {
                'floor_depth': config.floor_detection.floor_depth,
                'tolerance': config.floor_detection.tolerance,
                'object_height_offset': config.floor_detection.object_height_offset,
                'morphology_kernel_size': config.floor_detection.morphology_kernel_size,
                'close_kernel_size': config.floor_detection.close_kernel_size,
                'min_contour_area': config.floor_detection.min_contour_area,
                'max_contour_area': config.floor_detection.max_contour_area,
                'max_compactness': config.floor_detection.max_compactness,
                'median_blur_kernel': config.floor_detection.median_blur_kernel,
            },
            'calibration': {
                'target_width': config.calibration.target_width,
                'target_height': config.calibration.target_height,
                'calibration_file': config.calibration.calibration_file,
                'auto_load_calibration': config.calibration.auto_load_calibration,
                'auto_save_on_complete': config.calibration.auto_save_on_complete,
            },
            'tracking': {
                'max_feet_count': config.tracking.max_feet_count,
                'min_confidence': config.tracking.min_confidence,
                'tracking_history_size': config.tracking.tracking_history_size,
                'confidence_area_min': config.tracking.confidence_area_min,
                'confidence_area_max': config.tracking.confidence_area_max,
                'confidence_compactness_threshold': config.tracking.confidence_compactness_threshold,
                'confidence_depth_max': config.tracking.confidence_depth_max,
            },
            'visualization': {
                'show_trails': config.visualization.show_trails,
                'trail_length': config.visualization.trail_length,
                'trail_max_age': config.visualization.trail_max_age,
                'show_grid': config.visualization.show_grid,
                'grid_step': config.visualization.grid_step,
                'grid_major_step': config.visualization.grid_major_step,
                'show_zones': config.visualization.show_zones,
                'show_info_panel': config.visualization.show_info_panel,
                'foot_colors': config.visualization.foot_colors,
                'camera_foot_radius_base': config.visualization.camera_foot_radius_base,
                'camera_foot_radius_confidence_scale': config.visualization.camera_foot_radius_confidence_scale,
                'screen_foot_radius_base': config.visualization.screen_foot_radius_base,
            },
            'system': {
                'debug_mode': config.debug_mode,
                'verbose_logging': config.verbose_logging,
                'config_file': config.config_file,
                'log_file': config.log_file,
                'target_fps': config.target_fps,
                'performance_monitoring': config.performance_monitoring,
            }
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Преобразование словаря в конфигурацию"""
        camera_dict = config_dict.get('camera', {})
        camera_config = CameraConfig(
            width=camera_dict.get('width', 640),
            height=camera_dict.get('height', 480),
            fps=camera_dict.get('fps', 30),
            enable_decimation=camera_dict.get('enable_decimation', True),
            enable_spatial=camera_dict.get('enable_spatial', True),
            enable_temporal=camera_dict.get('enable_temporal', True),
            enable_hole_filling=camera_dict.get('enable_hole_filling', True),
        )
        
        floor_dict = config_dict.get('floor_detection', {})
        floor_config = FloorDetectionConfig(
            floor_depth=floor_dict.get('floor_depth', 1.5),
            tolerance=floor_dict.get('tolerance', 0.2),
            object_height_offset=floor_dict.get('object_height_offset', 0.05),
            morphology_kernel_size=floor_dict.get('morphology_kernel_size', 5),
            close_kernel_size=floor_dict.get('close_kernel_size', 15),
            min_contour_area=floor_dict.get('min_contour_area', 500),
            max_contour_area=floor_dict.get('max_contour_area', 50000),
            max_compactness=floor_dict.get('max_compactness', 50.0),
            median_blur_kernel=floor_dict.get('median_blur_kernel', 5),
        )
        
        calib_dict = config_dict.get('calibration', {})
        calib_config = CalibrationConfig(
            target_width=calib_dict.get('target_width', 1920),
            target_height=calib_dict.get('target_height', 1080),
            calibration_file=calib_dict.get('calibration_file', "calibration.json"),
            auto_load_calibration=calib_dict.get('auto_load_calibration', True),
            auto_save_on_complete=calib_dict.get('auto_save_on_complete', True),
        )
        
        track_dict = config_dict.get('tracking', {})
        track_config = TrackingConfig(
            max_feet_count=track_dict.get('max_feet_count', 10),
            min_confidence=track_dict.get('min_confidence', 0.3),
            tracking_history_size=track_dict.get('tracking_history_size', 10),
            confidence_area_min=track_dict.get('confidence_area_min', 500.0),
            confidence_area_max=track_dict.get('confidence_area_max', 20000.0),
            confidence_compactness_threshold=track_dict.get('confidence_compactness_threshold', 50.0),
            confidence_depth_max=track_dict.get('confidence_depth_max', 5.0),
        )
        
        vis_dict = config_dict.get('visualization', {})
        vis_config = VisualizationConfig(
            show_trails=vis_dict.get('show_trails', True),
            trail_length=vis_dict.get('trail_length', 30),
            trail_max_age=vis_dict.get('trail_max_age', 5.0),
            show_grid=vis_dict.get('show_grid', True),
            grid_step=vis_dict.get('grid_step', 100),
            grid_major_step=vis_dict.get('grid_major_step', 500),
            show_zones=vis_dict.get('show_zones', False),
            show_info_panel=vis_dict.get('show_info_panel', True),
            foot_colors=vis_dict.get('foot_colors', None),
            camera_foot_radius_base=vis_dict.get('camera_foot_radius_base', 8),
            camera_foot_radius_confidence_scale=vis_dict.get('camera_foot_radius_confidence_scale', 10),
            screen_foot_radius_base=vis_dict.get('screen_foot_radius_base', 30),
        )
        
        sys_dict = config_dict.get('system', {})
        
        return SystemConfig(
            camera=camera_config,
            floor_detection=floor_config,
            calibration=calib_config,
            tracking=track_config,
            visualization=vis_config,
            debug_mode=sys_dict.get('debug_mode', True),
            verbose_logging=sys_dict.get('verbose_logging', True),
            config_file=sys_dict.get('config_file', "system_config.json"),
            log_file=sys_dict.get('log_file', "foot_tracking.log"),
            target_fps=sys_dict.get('target_fps', 30.0),
            performance_monitoring=sys_dict.get('performance_monitoring', True),
        )
    
    def get_camera_config(self) -> CameraConfig:
        """Получение конфигурации камеры"""
        return self.config.camera
    
    def get_floor_detection_config(self) -> FloorDetectionConfig:
        """Получение конфигурации детекции пола"""
        return self.config.floor_detection
    
    def get_calibration_config(self) -> CalibrationConfig:
        """Получение конфигурации калибровки"""
        return self.config.calibration
    
    def get_tracking_config(self) -> TrackingConfig:
        """Получение конфигурации трекинга"""
        return self.config.tracking
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Получение конфигурации визуализации"""
        return self.config.visualization
    
    def update_floor_depth(self, depth: float) -> None:
        """Обновление глубины пола"""
        self.config.floor_detection.floor_depth = depth
    
    def update_target_resolution(self, width: int, height: int) -> None:
        """Обновление целевого разрешения"""
        self.config.calibration.target_width = width
        self.config.calibration.target_height = height


# Создание глобального экземпляра менеджера конфигурации
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """Получение текущей конфигурации"""
    return config_manager.config


def save_config(filename: str = None) -> bool:
    """Сохранение текущей конфигурации"""
    return config_manager.save_config(filename)


def load_config(filename: str = None) -> bool:
    """Загрузка конфигурации"""
    return config_manager.load_config(filename)


if __name__ == "__main__":
    # Тестирование конфигурации
    print("Тестирование системы конфигурации...")
    
    # Сохранение конфигурации по умолчанию
    config_manager.save_config("default_config.json")
    
    # Изменение некоторых параметров
    config_manager.config.floor_detection.floor_depth = 1.8
    config_manager.config.calibration.target_width = 1280
    config_manager.config.calibration.target_height = 720
    
    # Сохранение измененной конфигурации
    config_manager.save_config("test_config.json")
    
    # Загрузка исходной конфигурации
    config_manager.load_config("default_config.json")
    
    print(f"Глубина пола: {config_manager.config.floor_detection.floor_depth}")
    print(f"Целевое разрешение: {config_manager.config.calibration.target_width}x{config_manager.config.calibration.target_height}")
    
    print("Тестирование завершено!")
