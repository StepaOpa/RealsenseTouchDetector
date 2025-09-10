#!/usr/bin/env python3
"""
Демонстрация работы TouchFilter
Показывает как фильтр устраняет ложные касания и стабилизирует движения
"""

import numpy as np
import time
from touch_filter import TouchFilter


def simulate_touch_sequence():
    """Симуляция последовательности касаний с шумом"""
    print("🎯 Демонстрация TouchFilter - устранение ложных касаний")
    print("=" * 60)
    
    # Создаем фильтр с настройками по умолчанию
    touch_filter = TouchFilter(
        distance_threshold=30.0,
        history_size=5,
        min_movement_threshold=10.0
    )
    
    # Симуляция различных сценариев
    scenarios = [
        {
            "name": "Стабильное касание с шумом",
            "touches": [
                [(100, 100, {"confidence": 0.9})],  # Основное касание
                [(101, 101, {"confidence": 0.9})],  # Небольшой шум
                [(99, 102, {"confidence": 0.9})],   # Еще шум
                [(100, 100, {"confidence": 0.9})],  # Возврат к исходной позиции
                [(102, 99, {"confidence": 0.9})],   # Опять шум
            ]
        },
        {
            "name": "Значительное движение",
            "touches": [
                [(100, 100, {"confidence": 0.9})],  # Начальная позиция
                [(150, 150, {"confidence": 0.9})],  # Большое движение
                [(200, 200, {"confidence": 0.9})],  # Еще больше
                [(250, 250, {"confidence": 0.9})],  # Продолжение движения
            ]
        },
        {
            "name": "Множественные касания",
            "touches": [
                [(100, 100, {"confidence": 0.9}), (300, 300, {"confidence": 0.8})],  # Два касания
                [(101, 101, {"confidence": 0.9}), (301, 301, {"confidence": 0.8})],  # Небольшой шум
                [(102, 102, {"confidence": 0.9}), (302, 302, {"confidence": 0.8})],  # Еще шум
                [(150, 150, {"confidence": 0.9}), (350, 350, {"confidence": 0.8})],  # Значительное движение
            ]
        },
        {
            "name": "Исчезновение и появление касаний",
            "touches": [
                [(100, 100, {"confidence": 0.9})],  # Касание появляется
                [(101, 101, {"confidence": 0.9})],  # Небольшое движение
                [],                                  # Касание исчезает
                [(200, 200, {"confidence": 0.9})],  # Новое касание в другом месте
                [(201, 201, {"confidence": 0.9})],  # Небольшое движение
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Сценарий: {scenario['name']}")
        print("-" * 40)
        
        # Сбрасываем фильтр для каждого сценария
        touch_filter.reset()
        
        for frame_idx, touches in enumerate(scenario['touches']):
            # Применяем фильтр
            filtered_touches = touch_filter.update(touches)
            
            # Выводим результаты
            print(f"Кадр {frame_idx + 1}:")
            print(f"  Входные касания: {len(touches)}")
            for i, (x, y, info) in enumerate(touches):
                print(f"    Touch {i}: ({x}, {y}) conf={info['confidence']:.1f}")
            
            print(f"  Отфильтрованные: {len(filtered_touches)}")
            for i, (x, y, info) in enumerate(filtered_touches):
                print(f"    Touch {i}: ({x}, {y}) conf={info['confidence']:.1f}")
            
            # Показываем статистику фильтра
            stats = touch_filter.get_statistics()
            print(f"  Статистика: {stats['total_filtered']}/{stats['total_processed']} касаний прошли фильтр")
            
            time.sleep(0.1)  # Небольшая задержка для наглядности
    
    print("\n" + "=" * 60)
    print("📊 Итоговая статистика фильтра:")
    final_stats = touch_filter.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


def test_filter_parameters():
    """Тестирование различных параметров фильтра"""
    print("\n🔧 Тестирование параметров TouchFilter")
    print("=" * 60)
    
    # Тестовые данные - касание с постепенным движением
    test_touches = [
        [(100, 100, {"confidence": 0.9})],
        [(105, 105, {"confidence": 0.9})],  # 7px движение
        [(110, 110, {"confidence": 0.9})],  # 7px движение
        [(115, 115, {"confidence": 0.9})],  # 7px движение
        [(120, 120, {"confidence": 0.9})],  # 7px движение
    ]
    
    # Тестируем разные пороги минимального движения
    thresholds = [5, 10, 15, 20]
    
    for threshold in thresholds:
        print(f"\n🎯 Тест с минимальным движением: {threshold}px")
        
        filter_test = TouchFilter(
            distance_threshold=30.0,
            history_size=5,
            min_movement_threshold=threshold
        )
        
        for frame_idx, touches in enumerate(test_touches):
            filtered = filter_test.update(touches)
            print(f"  Кадр {frame_idx + 1}: {len(touches)} -> {len(filtered)} касаний")
        
        stats = filter_test.get_statistics()
        print(f"  Результат: {stats['total_filtered']}/{stats['total_processed']} касаний прошли фильтр")


def performance_test():
    """Тест производительности фильтра"""
    print("\n⚡ Тест производительности TouchFilter")
    print("=" * 60)
    
    import time
    
    # Создаем фильтр
    touch_filter = TouchFilter()
    
    # Генерируем много случайных касаний
    num_frames = 1000
    touches_per_frame = 5
    
    print(f"Тестируем {num_frames} кадров по {touches_per_frame} касаний...")
    
    start_time = time.time()
    
    for frame in range(num_frames):
        # Генерируем случайные касания
        touches = []
        for i in range(touches_per_frame):
            x = np.random.randint(0, 800)
            y = np.random.randint(0, 600)
            confidence = np.random.uniform(0.5, 1.0)
            touches.append((x, y, {"confidence": confidence}))
        
        # Применяем фильтр
        filtered = touch_filter.update(touches)
    
    end_time = time.time()
    
    # Результаты
    total_time = end_time - start_time
    fps = num_frames / total_time
    avg_time_per_frame = total_time / num_frames * 1000  # в миллисекундах
    
    print(f"✅ Результаты производительности:")
    print(f"  Общее время: {total_time:.3f} секунд")
    print(f"  FPS: {fps:.1f}")
    print(f"  Время на кадр: {avg_time_per_frame:.2f} мс")
    
    stats = touch_filter.get_statistics()
    print(f"  Обработано касаний: {stats['total_processed']}")
    print(f"  Отфильтровано касаний: {stats['total_filtered']}")
    print(f"  Эффективность фильтрации: {stats['filter_ratio']:.1f}%")


def main():
    """Основная функция демонстрации"""
    print("🧪 Демонстрация TouchFilter - продвинутая фильтрация касаний")
    print("🎯 Цель: устранение ложных касаний и стабилизация движений")
    print()
    
    try:
        # Основная демонстрация
        simulate_touch_sequence()
        
        # Тестирование параметров
        test_filter_parameters()
        
        # Тест производительности
        performance_test()
        
        print("\n✅ Демонстрация завершена!")
        print("💡 TouchFilter успешно интегрирован в TouchProcessor")
        print("🎛️ Управление доступно через ползунки в окне настроек")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()
