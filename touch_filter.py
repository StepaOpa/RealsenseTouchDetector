"""
Модуль для продвинутой фильтрации касаний
Реализует алгоритм отслеживания и фильтрации касаний с использованием истории позиций
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class TouchPosition:
    """Структура для хранения позиции касания"""
    x: float
    y: float
    timestamp: float
    confidence: float = 1.0


class TouchFilter:
    """
    Продвинутый фильтр касаний с отслеживанием истории позиций
    Фильтрует касания на основе значительных изменений позиции
    """
    
    def __init__(self, distance_threshold: float = 30.0, history_size: int = 5, 
                 min_movement_threshold: float = 10.0, timeout: float = 2.0) -> None:
        """
        Инициализация фильтра касаний
        
        Args:
            distance_threshold: Пороговое расстояние в пикселях для определения значительного изменения
            history_size: Количество предыдущих позиций для хранения в истории
            min_movement_threshold: Минимальное движение для регистрации изменения
            timeout: Время в секундах, после которого статичное касание скрывается
        """
        self.distance_threshold: float = distance_threshold
        self.history_size: int = history_size
        self.min_movement_threshold: float = min_movement_threshold
        self.timeout: float = timeout
        
        # История касаний: {touch_id: [TouchPosition, ...]}
        self.touch_history: Dict[int, List[TouchPosition]] = {}
        self.next_touch_id: int = 0
        
        # Временные метки для таймаута
        self.touch_timestamps: Dict[int, float] = {}  # Время последнего движения
        self.touch_start_time: Dict[int, float] = {}  # Время первого появления
        
        # Настройка логирования
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Статистика
        self.total_touches_processed: int = 0
        self.total_touches_filtered: int = 0
        self.total_touches_tracked: int = 0
        
        self.logger.info(f"TouchFilter инициализирован: threshold={distance_threshold}px, history={history_size}")
    
    def update(self, current_touches: List[Tuple[int, int, Dict]]) -> List[Tuple[int, int, Dict]]:
        """
        Обновление состояния фильтра и возврат отфильтрованных касаний
        
        Args:
            current_touches: Список текущих касаний в формате [(x, y, info), ...]
            
        Returns:
            filtered_touches: Отфильтрованные касания, которые значительно изменились
        """
        import time
        current_time = time.time()
        
        self.total_touches_processed += len(current_touches)
        
        # Если нет текущих касаний, очищаем историю
        if not current_touches:
            self._cleanup_old_touches(current_time)
            return []
        
        filtered_touches = []
        used_history_ids: Set[int] = set()
        
        # Сопоставляем текущие касания с историей
        for touch in current_touches:
            x, y, info = touch
            touch_pos = np.array([x, y])
            confidence = info.get('confidence', 1.0)
            
            matched = False
            
            # Ищем ближайшее касание в истории
            for touch_id, history in self.touch_history.items():
                if touch_id in used_history_ids:
                    continue
                
                # Берем последнюю позицию из истории
                last_pos = history[-1]
                last_pos_array = np.array([last_pos.x, last_pos.y])
                distance = np.linalg.norm(touch_pos - last_pos_array)
                
                # Если расстояние меньше порога, считаем это тем же касанием
                if distance < self.distance_threshold:
                    # Проверяем, было ли значительное движение
                    if distance > self.min_movement_threshold:
                        # Было значительное движение - обновляем время последнего движения
                        self.touch_timestamps[touch_id] = current_time
                    
                    # Обновляем историю для этого касания
                    new_position = TouchPosition(x, y, current_time, confidence)
                    self.touch_history[touch_id].append(new_position)
                    
                    # Ограничиваем размер истории
                    if len(self.touch_history[touch_id]) > self.history_size:
                        self.touch_history[touch_id].pop(0)
                    
                    # Проверяем таймаут - касание активно только если не превышен таймаут
                    time_since_movement = current_time - self.touch_timestamps.get(touch_id, current_time)
                    if time_since_movement <= self.timeout:
                        # Добавляем в отфильтрованные касания
                        filtered_touches.append(touch)
                        self.total_touches_filtered += 1
                    
                    used_history_ids.add(touch_id)
                    matched = True
                    break
            
            # Если не нашли соответствия в истории, это новое касание
            if not matched:
                new_id = self.next_touch_id
                self.next_touch_id += 1
                
                new_position = TouchPosition(x, y, current_time, confidence)
                self.touch_history[new_id] = [new_position]
                self.touch_timestamps[new_id] = current_time
                self.touch_start_time[new_id] = current_time
                
                # Новые касания всегда добавляем
                filtered_touches.append(touch)
                self.total_touches_filtered += 1
                self.total_touches_tracked += 1
        
        # Удаляем касания, которые исчезли
        self._remove_disappeared_touches(used_history_ids, current_time)
        
        return filtered_touches
    
    def _is_significant_movement(self, touch_id: int, new_pos: np.ndarray, current_time: float) -> bool:
        """
        Проверяет, является ли движение касания значительным
        
        Args:
            touch_id: ID касания
            new_pos: Новая позиция
            current_time: Текущее время
            
        Returns:
            True если движение значительное
        """
        if touch_id not in self.touch_history:
            return True  # Новое касание всегда значительное
        
        history = self.touch_history[touch_id]
        if len(history) < 2:
            return True  # Недостаточно истории для сравнения
        
        # Берем последнюю позицию
        last_pos = history[-1]
        last_pos_array = np.array([last_pos.x, last_pos.y])
        
        # Вычисляем расстояние
        distance = np.linalg.norm(new_pos - last_pos_array)
        
        # Проверяем минимальное движение
        if distance < self.min_movement_threshold:
            return False
        
        # Проверяем скорость движения (опционально)
        time_diff = current_time - last_pos.timestamp
        if time_diff > 0:
            velocity = distance / time_diff
            # Если скорость слишком высокая, возможно это шум
            if velocity > 1000:  # пикселей в секунду
                return False
        
        return True
    
    def _cleanup_old_touches(self, current_time: float) -> None:
        """Очистка старых касаний из истории"""
        max_age = 2.0  # Максимальный возраст касания в секундах
        
        touch_ids_to_remove = []
        for touch_id, history in self.touch_history.items():
            if history and current_time - history[-1].timestamp > max_age:
                touch_ids_to_remove.append(touch_id)
        
        for touch_id in touch_ids_to_remove:
            del self.touch_history[touch_id]
    
    def _remove_disappeared_touches(self, used_ids: Set[int], current_time: float) -> None:
        """Удаление касаний, которые исчезли или превысили таймаут"""
        touch_ids_to_remove = []
        
        for touch_id in self.touch_history.keys():
            if touch_id not in used_ids:
                # Касание исчезло из текущего кадра
                touch_ids_to_remove.append(touch_id)
            else:
                # Проверяем таймаут для активных касаний
                if touch_id in self.touch_timestamps:
                    time_since_movement = current_time - self.touch_timestamps[touch_id]
                    if time_since_movement > self.timeout:
                        # Касание превысило таймаут без движения
                        touch_ids_to_remove.append(touch_id)
        
        # Удаляем касания
        for touch_id in touch_ids_to_remove:
            if touch_id in self.touch_history:
                del self.touch_history[touch_id]
            if touch_id in self.touch_timestamps:
                del self.touch_timestamps[touch_id]
            if touch_id in self.touch_start_time:
                del self.touch_start_time[touch_id]
    
    def set_threshold(self, threshold: float) -> None:
        """
        Установка нового порогового значения
        
        Args:
            threshold: Новое пороговое расстояние в пикселях
        """
        self.distance_threshold = max(5.0, min(100.0, threshold))
        self.logger.info(f"Порог фильтрации изменен на: {self.distance_threshold}px")
    
    def set_min_movement(self, min_movement: float) -> None:
        """
        Установка минимального порога движения
        
        Args:
            min_movement: Минимальное движение в пикселях
        """
        self.min_movement_threshold = max(1.0, min(50.0, min_movement))
        self.logger.info(f"Минимальное движение изменено на: {self.min_movement_threshold}px")
    
    def set_history_size(self, history_size: int) -> None:
        """
        Установка размера истории
        
        Args:
            history_size: Количество позиций в истории
        """
        self.history_size = max(2, min(20, history_size))
        self.logger.info(f"Размер истории изменен на: {self.history_size}")
    
    def set_timeout(self, timeout: float) -> None:
        """
        Установка таймаута для статичных касаний
        
        Args:
            timeout: Время в секундах, после которого статичное касание скрывается
        """
        self.timeout = max(0.1, min(10.0, timeout))
        self.logger.info(f"Таймаут касаний изменен на: {self.timeout}с")
    
    def get_timeout(self) -> float:
        """Возвращает текущий таймаут"""
        return self.timeout
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Получение статистики фильтра
        
        Returns:
            Словарь со статистикой
        """
        return {
            "total_processed": self.total_touches_processed,
            "total_filtered": self.total_touches_filtered,
            "total_tracked": self.total_touches_tracked,
            "active_touches": len(self.touch_history),
            "filter_ratio": (self.total_touches_filtered / max(1, self.total_touches_processed)) * 100,
            "distance_threshold": self.distance_threshold,
            "min_movement_threshold": self.min_movement_threshold,
            "history_size": self.history_size,
            "timeout": self.timeout,
            "timed_out_touches": len(self.touch_timestamps) - len(self.touch_history)
        }
    
    def reset(self) -> None:
        """Сброс состояния фильтра"""
        self.touch_history.clear()
        self.touch_timestamps.clear()
        self.touch_start_time.clear()
        self.next_touch_id = 0
        self.total_touches_processed = 0
        self.total_touches_filtered = 0
        self.total_touches_tracked = 0
        self.logger.info("Состояние фильтра сброшено")
    
    def get_active_touch_positions(self) -> List[Tuple[float, float]]:
        """
        Получение позиций активных касаний
        
        Returns:
            Список позиций активных касаний
        """
        positions = []
        for history in self.touch_history.values():
            if history:
                last_pos = history[-1]
                positions.append((last_pos.x, last_pos.y))
        return positions
    
    def get_touch_durations(self, current_time: float = None) -> Dict[int, float]:
        """
        Получение длительности каждого активного касания
        
        Args:
            current_time: Текущее время (если None, используется time.time())
            
        Returns:
            Словарь {touch_id: duration_in_seconds}
        """
        if current_time is None:
            import time
            current_time = time.time()
        
        durations = {}
        for touch_id, start_time in self.touch_start_time.items():
            if touch_id in self.touch_history:  # Касание все еще активно
                durations[touch_id] = current_time - start_time
        return durations


def test_touch_filter() -> None:
    """Тест фильтра касаний"""
    print("🧪 Тестирование TouchFilter")
    
    # Создаем фильтр
    touch_filter = TouchFilter(distance_threshold=30, history_size=5)
    
    # Симуляция последовательности касаний
    touch_sequences = [
        [(100, 100, {"confidence": 0.9})],  # Первое касание
        [(101, 101, {"confidence": 0.9})],  # Почти то же положение
        [(102, 102, {"confidence": 0.9})],  # Все еще близко
        [(150, 150, {"confidence": 0.9})],  # Значительное изменение
        [(151, 151, {"confidence": 0.9})],  # Опять близко
        [],                                  # Касаний нет
        [(200, 200, {"confidence": 0.9})]   # Новое касание
    ]
    
    for i, touches in enumerate(touch_sequences):
        filtered = touch_filter.update(touches)
        print(f"Кадр {i}: {len(touches)} касаний -> {len(filtered)} отфильтровано")
        if filtered:
            for j, (x, y, info) in enumerate(filtered):
                print(f"  Touch {j}: ({x}, {y}) conf={info['confidence']}")
    
    # Показываем статистику
    stats = touch_filter.get_statistics()
    print(f"\n📊 Статистика фильтра:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_touch_filter()
