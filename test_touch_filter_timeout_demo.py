#!/usr/bin/env python3
"""
Демонстрация TouchFilter с функциональностью таймаута
Показывает как фильтр автоматически скрывает статичные касания
"""

import numpy as np
import time
from touch_filter import TouchFilter


def simulate_static_touch_scenario():
    """Симуляция статичного касания с таймаутом"""
    print("⏰ Демонстрация TouchFilter с таймаутом - скрытие статичных касаний")
    print("=" * 70)
    
    # Создаем фильтр с таймаутом 2 секунды
    touch_filter = TouchFilter(
        distance_threshold=30.0,
        history_size=5,
        min_movement_threshold=10.0,
        timeout=2.0
    )
    
    print("🎯 Сценарий: Статичное касание с таймаутом 2 секунды")
    print("📝 Ожидаемое поведение:")
    print("   - Касание появляется и регистрируется")
    print("   - Небольшие движения не сбрасывают таймер")
    print("   - Через 2 секунды без движения касание исчезает")
    print("   - Новое движение сбрасывает таймер")
    print()
    
    # Симуляция статичного касания
    touch_sequences = [
        # Фаза 1: Появление касания
        {"touches": [(100, 100, {"confidence": 0.9})], "delay": 0.5, "description": "Касание появляется"},
        {"touches": [(101, 101, {"confidence": 0.9})], "delay": 0.5, "description": "Небольшое движение"},
        {"touches": [(102, 102, {"confidence": 0.9})], "delay": 0.5, "description": "Еще небольшое движение"},
        
        # Фаза 2: Статичное состояние (должно привести к таймауту)
        {"touches": [(102, 102, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 1"},
        {"touches": [(102, 102, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 2"},
        {"touches": [(102, 102, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 3"},
        {"touches": [(102, 102, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 4"},
        {"touches": [(102, 102, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 5"},
        
        # Фаза 3: Новое движение (сброс таймера)
        {"touches": [(150, 150, {"confidence": 0.9})], "delay": 0.5, "description": "Значительное движение - сброс таймера"},
        {"touches": [(151, 151, {"confidence": 0.9})], "delay": 0.5, "description": "Небольшое движение"},
        
        # Фаза 4: Снова статичное состояние
        {"touches": [(151, 151, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 1"},
        {"touches": [(151, 151, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 2"},
        {"touches": [(151, 151, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 3"},
        {"touches": [(151, 151, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 4"},
        {"touches": [(151, 151, {"confidence": 0.9})], "delay": 0.5, "description": "Статичное состояние 5"},
    ]
    
    for i, frame in enumerate(touch_sequences):
        print(f"📋 Кадр {i + 1}: {frame['description']}")
        
        # Применяем фильтр
        filtered_touches = touch_filter.update(frame['touches'])
        
        # Выводим результаты
        print(f"   Входные касания: {len(frame['touches'])}")
        for j, (x, y, info) in enumerate(frame['touches']):
            print(f"     Touch {j}: ({x}, {y}) conf={info['confidence']:.1f}")
        
        print(f"   Отфильтрованные: {len(filtered_touches)}")
        for j, (x, y, info) in enumerate(filtered_touches):
            print(f"     Touch {j}: ({x}, {y}) conf={info['confidence']:.1f}")
        
        # Показываем информацию о времени
        current_time = time.time()
        durations = touch_filter.get_touch_durations(current_time)
        for touch_id, duration in durations.items():
            print(f"   Касание {touch_id}: длительность {duration:.1f}с")
        
        # Показываем статистику
        stats = touch_filter.get_statistics()
        print(f"   Активных касаний: {stats['active_touches']}")
        print(f"   Таймаут: {stats['timeout']}с")
        
        print()
        
        # Задержка между кадрами
        time.sleep(frame['delay'])
    
    print("✅ Демонстрация статичного касания завершена!")


def test_different_timeouts():
    """Тестирование различных значений таймаута"""
    print("\n🔧 Тестирование различных значений таймаута")
    print("=" * 70)
    
    timeouts = [0.5, 1.0, 2.0, 5.0]
    
    for timeout in timeouts:
        print(f"\n⏱️ Тест с таймаутом: {timeout}с")
        
        # Создаем фильтр с текущим таймаутом
        touch_filter = TouchFilter(
            distance_threshold=30.0,
            history_size=5,
            min_movement_threshold=10.0,
            timeout=timeout
        )
        
        # Симуляция статичного касания
        static_touches = [(100, 100, {"confidence": 0.9})]
        
        # Применяем касание несколько раз с задержкой
        for i in range(int(timeout * 2) + 2):  # В 2 раза больше таймаута
            filtered = touch_filter.update(static_touches)
            stats = touch_filter.get_statistics()
            
            print(f"  Кадр {i + 1}: {len(filtered)} касаний, активных: {stats['active_touches']}")
            
            if len(filtered) == 0:
                print(f"  ✅ Касание исчезло через {i + 1} кадров (ожидалось ~{timeout * 2})")
                break
            
            time.sleep(0.5)  # 0.5 секунды между кадрами
        else:
            print(f"  ⚠️ Касание не исчезло в течение {timeout * 2} секунд")


def test_movement_resets_timeout():
    """Тестирование сброса таймаута при движении"""
    print("\n🔄 Тестирование сброса таймаута при движении")
    print("=" * 70)
    
    # Создаем фильтр с коротким таймаутом
    touch_filter = TouchFilter(
        distance_threshold=30.0,
        history_size=5,
        min_movement_threshold=10.0,
        timeout=1.0  # 1 секунда
    )
    
    print("🎯 Сценарий: Движение сбрасывает таймаут")
    print("📝 Ожидаемое поведение:")
    print("   - Касание не должно исчезнуть, если есть движение")
    print("   - Таймаут сбрасывается при значительном движении")
    print()
    
    # Симуляция с периодическими движениями
    touch_sequences = [
        # Начальное касание
        {"touches": [(100, 100, {"confidence": 0.9})], "delay": 0.3, "description": "Начальное касание"},
        {"touches": [(100, 100, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 1"},
        {"touches": [(100, 100, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 2"},
        
        # Движение (сброс таймаута)
        {"touches": [(120, 120, {"confidence": 0.9})], "delay": 0.3, "description": "Движение - сброс таймаута"},
        {"touches": [(120, 120, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 1"},
        {"touches": [(120, 120, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 2"},
        
        # Еще движение
        {"touches": [(140, 140, {"confidence": 0.9})], "delay": 0.3, "description": "Движение - сброс таймаута"},
        {"touches": [(140, 140, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 1"},
        {"touches": [(140, 140, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 2"},
        {"touches": [(140, 140, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 3"},
        {"touches": [(140, 140, {"confidence": 0.9})], "delay": 0.3, "description": "Статичное 4"},
    ]
    
    for i, frame in enumerate(touch_sequences):
        print(f"📋 Кадр {i + 1}: {frame['description']}")
        
        # Применяем фильтр
        filtered_touches = touch_filter.update(frame['touches'])
        
        # Выводим результаты
        print(f"   Отфильтрованные: {len(filtered_touches)}")
        
        # Показываем статистику
        stats = touch_filter.get_statistics()
        print(f"   Активных касаний: {stats['active_touches']}")
        
        if len(filtered_touches) == 0:
            print(f"   ⏰ Касание исчезло на кадре {i + 1}")
            break
        
        print()
        time.sleep(frame['delay'])
    
    print("✅ Тестирование сброса таймаута завершено!")


def test_multiple_touches_timeout():
    """Тестирование таймаута для множественных касаний"""
    print("\n👥 Тестирование таймаута для множественных касаний")
    print("=" * 70)
    
    # Создаем фильтр
    touch_filter = TouchFilter(
        distance_threshold=30.0,
        history_size=5,
        min_movement_threshold=10.0,
        timeout=1.5  # 1.5 секунды
    )
    
    print("🎯 Сценарий: Множественные касания с разными таймаутами")
    print("📝 Ожидаемое поведение:")
    print("   - Каждое касание имеет свой таймаут")
    print("   - Касания исчезают независимо друг от друга")
    print()
    
    # Симуляция множественных касаний
    touch_sequences = [
        # Появление двух касаний
        {"touches": [(100, 100, {"confidence": 0.9}), (200, 200, {"confidence": 0.9})], 
         "delay": 0.5, "description": "Два касания появляются"},
        
        # Первое касание движется, второе статично
        {"touches": [(110, 110, {"confidence": 0.9}), (200, 200, {"confidence": 0.9})], 
         "delay": 0.5, "description": "Первое движется, второе статично"},
        {"touches": [(110, 110, {"confidence": 0.9}), (200, 200, {"confidence": 0.9})], 
         "delay": 0.5, "description": "Первое статично, второе статично"},
        {"touches": [(110, 110, {"confidence": 0.9}), (200, 200, {"confidence": 0.9})], 
         "delay": 0.5, "description": "Первое статично, второе статично"},
        
        # Второе касание должно исчезнуть
        {"touches": [(110, 110, {"confidence": 0.9}), (200, 200, {"confidence": 0.9})], 
         "delay": 0.5, "description": "Второе касание должно исчезнуть"},
        
        # Первое касание движется, сбрасывая таймаут
        {"touches": [(130, 130, {"confidence": 0.9})], 
         "delay": 0.5, "description": "Только первое касание (второе исчезло)"},
    ]
    
    for i, frame in enumerate(touch_sequences):
        print(f"📋 Кадр {i + 1}: {frame['description']}")
        
        # Применяем фильтр
        filtered_touches = touch_filter.update(frame['touches'])
        
        # Выводим результаты
        print(f"   Входных касаний: {len(frame['touches'])}")
        print(f"   Отфильтрованных: {len(filtered_touches)}")
        
        for j, (x, y, info) in enumerate(filtered_touches):
            print(f"     Touch {j}: ({x}, {y}) conf={info['confidence']:.1f}")
        
        # Показываем статистику
        stats = touch_filter.get_statistics()
        print(f"   Активных касаний: {stats['active_touches']}")
        
        print()
        time.sleep(frame['delay'])
    
    print("✅ Тестирование множественных касаний завершено!")


def main():
    """Основная функция демонстрации"""
    print("⏰ Демонстрация TouchFilter с функциональностью таймаута")
    print("🎯 Цель: автоматическое скрытие статичных касаний")
    print()
    
    try:
        # Основная демонстрация статичного касания
        simulate_static_touch_scenario()
        
        # Тестирование различных таймаутов
        test_different_timeouts()
        
        # Тестирование сброса таймаута при движении
        test_movement_resets_timeout()
        
        # Тестирование множественных касаний
        test_multiple_touches_timeout()
        
        print("\n" + "=" * 70)
        print("✅ Демонстрация TouchFilter с таймаутом завершена!")
        print("💡 Функциональность таймаута успешно интегрирована")
        print("🎛️ Управление доступно через ползунок 'touch timeout' в окне настроек")
        print("⏱️ Диапазон: 0.1 - 10.0 секунд")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()
