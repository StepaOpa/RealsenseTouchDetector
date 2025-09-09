"""
Модуль для UDP-коммуникации с Unity в реальном времени
Отправляет координаты касаний и другие данные детекции
"""

import socket
import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class TouchData:
    """Структура данных касания для отправки в Unity"""
    id: int
    x: float          # Нормализованная координата X (0.0-1.0)
    y: float          # Нормализованная координата Y (0.0-1.0)
    depth: float      # Глубина касания в метрах
    area: int         # Площадь касания в пикселях
    confidence: float # Уверенность в касании (0.0-1.0)
    timestamp: float  # Время касания
    type: str = "touch"  # Тип события


class UnityCommunication:
    """
    Класс для UDP-коммуникации с Unity приложением
    Отправляет данные касаний в реальном времени
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8052, 
                 target_width: int = 1920, target_height: int = 1080) -> None:
        """
        Инициализация UDP коммуникации с Unity
        
        Args:
            host: IP адрес Unity приложения
            port: UDP порт для отправки данных
            target_width: Целевая ширина экрана для нормализации координат
            target_height: Целевая высота экрана для нормализации координат
        """
        self.host: str = host
        self.port: int = port
        self.target_width: int = target_width
        self.target_height: int = target_height
        
        # Настройка сокета
        self.sock: Optional[socket.socket] = None
        self.server_address: Tuple[str, int] = (host, port)
        self.is_connected: bool = False
        
        # Настройка логирования
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Статистика отправки
        self.total_sent: int = 0
        self.total_errors: int = 0
        self.last_send_time: float = 0.0
        
        # Настройки отправки
        self.send_enabled: bool = True
        self.max_touches_per_frame: int = 10  # Максимум касаний в одном кадре
        self.min_send_interval: float = 0.001  # Минимальный интервал между отправками (1мс)
        
        # Инициализация соединения
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Инициализация UDP соединения"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0)  # Таймаут 1 секунда
            self.is_connected = True
            self.logger.info(f"Unity коммуникация инициализирована: {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Unity коммуникации: {e}")
            self.is_connected = False
    
    def normalize_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Нормализация координат в диапазон 0.0-1.0
        
        Args:
            x: X координата в пикселях
            y: Y координата в пикселях
            
        Returns:
            Кортеж нормализованных координат (norm_x, norm_y)
        """
        norm_x = max(0.0, min(1.0, x / self.target_width))
        norm_y = max(0.0, min(1.0, y / self.target_height))
        return norm_x, norm_y
    
    def format_touches_for_unity(self, touches: List[Tuple[int, int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Форматирование данных касаний для Unity
        
        Args:
            touches: Список касаний из TouchProcessor в формате [(x, y, info), ...]
            
        Returns:
            Список словарей с данными касаний для Unity
        """
        unity_touches = []
        current_time = time.time()
        
        for i, (x, y, info) in enumerate(touches[:self.max_touches_per_frame]):
            # Нормализуем координаты
            norm_x, norm_y = self.normalize_coordinates(x, y)
            
            # Создаем структуру данных для Unity
            touch_data = {
                "id": i,
                "x": round(norm_x, 4),
                "y": round(norm_y, 4),
                "depth": round(info.get('depth', 0.0), 3),
                "area": info.get('area', 0),
                "confidence": round(info.get('confidence', 1.0), 3),
                "timestamp": round(current_time, 3),
                "type": "touch"
            }
            
            unity_touches.append(touch_data)
        
        return unity_touches
    
    def send_touch_coordinates(self, touches: List[Tuple[int, int, Dict[str, Any]]]) -> bool:
        """
        Отправка координат касаний в Unity
        
        Args:
            touches: Список касаний из TouchProcessor
            
        Returns:
            True если отправка успешна, False в случае ошибки
        """
        if not self.send_enabled or not self.is_connected or not touches:
            return False
        
        # Проверяем минимальный интервал между отправками
        current_time = time.time()
        if current_time - self.last_send_time < self.min_send_interval:
            return False
        
        try:
            # Форматируем данные для Unity
            unity_touches = self.format_touches_for_unity(touches)
            
            # Создаем полное сообщение
            message = {
                "type": "touch_data",
                "timestamp": round(current_time, 3),
                "touches": unity_touches,
                "touch_count": len(unity_touches),
                "frame_id": self.total_sent + 1
            }
            
            # Преобразуем в JSON и отправляем
            json_message = json.dumps(message, ensure_ascii=False)
            self.sock.sendto(json_message.encode('utf-8'), self.server_address)
            
            # Обновляем статистику
            self.total_sent += 1
            self.last_send_time = current_time
            
            self.logger.debug(f"Отправлено {len(unity_touches)} касаний в Unity")
            return True
            
        except Exception as e:
            self.total_errors += 1
            self.logger.error(f"Ошибка отправки касаний в Unity: {e}")
            return False
    
    def send_system_status(self, status: Dict[str, Any]) -> bool:
        """
        Отправка системного статуса в Unity
        
        Args:
            status: Словарь со статусной информацией
            
        Returns:
            True если отправка успешна
        """
        if not self.send_enabled or not self.is_connected:
            return False
        
        try:
            message = {
                "type": "system_status",
                "timestamp": round(time.time(), 3),
                "status": status
            }
            
            json_message = json.dumps(message, ensure_ascii=False)
            self.sock.sendto(json_message.encode('utf-8'), self.server_address)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки статуса в Unity: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Тестирование соединения с Unity
        
        Returns:
            True если соединение работает
        """
        test_message = {
            "type": "connection_test",
            "timestamp": round(time.time(), 3),
            "message": "Test connection from RealSense Touch Detector"
        }
        
        try:
            json_message = json.dumps(test_message)
            self.sock.sendto(json_message.encode('utf-8'), self.server_address)
            self.logger.info("Тестовое сообщение отправлено в Unity")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка тестирования соединения: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики отправки
        
        Returns:
            Словарь со статистикой
        """
        return {
            "total_sent": self.total_sent,
            "total_errors": self.total_errors,
            "success_rate": (self.total_sent / max(1, self.total_sent + self.total_errors)) * 100,
            "is_connected": self.is_connected,
            "server_address": f"{self.host}:{self.port}",
            "send_enabled": self.send_enabled
        }
    
    def set_enabled(self, enabled: bool) -> None:
        """Включение/выключение отправки данных"""
        self.send_enabled = enabled
        status = "включена" if enabled else "выключена"
        self.logger.info(f"Отправка данных в Unity {status}")
    
    def set_server_address(self, host: str, port: int) -> None:
        """
        Изменение адреса Unity сервера
        
        Args:
            host: Новый IP адрес
            port: Новый порт
        """
        self.host = host
        self.port = port
        self.server_address = (host, port)
        self.logger.info(f"Адрес Unity сервера изменен на {host}:{port}")
    
    def close(self) -> None:
        """Закрытие соединения"""
        if self.sock:
            try:
                self.sock.close()
                self.is_connected = False
                self.logger.info("Unity коммуникация закрыта")
            except Exception as e:
                self.logger.error(f"Ошибка закрытия соединения: {e}")


def test_unity_communication() -> None:
    """Тест Unity коммуникации"""
    print("🔌 Тестирование Unity коммуникации")
    
    # Создаем коммуникатор
    unity_comm = UnityCommunication()
    
    # Тестируем соединение
    if unity_comm.test_connection():
        print("✅ Соединение с Unity установлено")
    else:
        print("❌ Ошибка соединения с Unity")
    
    # Тестируем отправку касаний
    test_touches = [
        (960, 540, {"depth": 0.125, "area": 1500, "confidence": 0.95}),
        (1200, 300, {"depth": 0.108, "area": 800, "confidence": 0.87})
    ]
    
    if unity_comm.send_touch_coordinates(test_touches):
        print("✅ Тестовые касания отправлены")
    else:
        print("❌ Ошибка отправки касаний")
    
    # Показываем статистику
    stats = unity_comm.get_statistics()
    print(f"📊 Статистика: {stats}")
    
    unity_comm.close()


if __name__ == "__main__":
    test_unity_communication()
