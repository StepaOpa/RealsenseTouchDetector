#!/usr/bin/env python3
"""
Демонстрация методов коррекции глубины для устранения параллакса камера-проектор
"""

import numpy as np
import cv2
from touch_processor import TouchProcessor

def create_test_depth_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Создает тестовое изображение глубины"""
    # Создаем градиент глубины (имитация поверхности)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    base_depth = 1000 + (y * 2)  # От 1000мм до 2000мм
    
    # Добавляем несколько "объектов" (касаний)
    objects = np.zeros_like(base_depth)
    
    # Объект 1: круг в левом верхнем углу
    cv2.circle(objects, (160, 120), 30, -50, -1)  # 50мм ближе к камере
    
    # Объект 2: прямоугольник в центре
    cv2.rectangle(objects, (280, 200), (360, 280), -30, -1)  # 30мм ближе
    
    # Объект 3: круг в правом нижнем углу  
    cv2.circle(objects, (480, 360), 25, -40, -1)  # 40мм ближе
    
    # Добавляем шум
    noise = np.random.normal(0, 5, base_depth.shape)
    
    result = base_depth + objects + noise
    return np.clip(result, 0, 5000).astype(np.uint16)

def demonstrate_correction_methods():
    """Демонстрирует различные методы коррекции глубины"""
    print("🔧 Демонстрация методов коррекции глубины для устранения параллакса")
    print("=" * 70)
    
    # Создаем процессор касаний
    processor = TouchProcessor(640, 480, 1920, 1080)
    
    # Создаем тестовое изображение глубины
    test_depth = create_test_depth_image()
    print(f"📷 Создано тестовое изображение глубины: {test_depth.shape}")
    print(f"   Диапазон глубины: {test_depth.min()}мм - {test_depth.max()}мм")
    
    # Демонстрируем различные методы коррекции
    corrections = [
        ("Исходное изображение", lambda img: img),
        ("Смещение +50мм", lambda img: processor.adjust_depth_perception(img, 50.0)),
        ("Масштабирование 1.1x", lambda img: processor.scale_depth_perception(img, 1.1)),
        ("Медианная фильтрация", lambda img: processor.spatial_depth_filter(img, 5)),
        ("Комплексная коррекция", lambda img: processor.virtual_camera_distance_adjustment(img)),
    ]
    
    # Настраиваем параметры для комплексной коррекции
    processor.depth_offset_mm = 25.0
    processor.depth_scale_factor = 1.05
    processor.enable_spatial_filter = True
    processor.spatial_filter_kernel = 5
    
    print("\n🎛️ Применяем различные методы коррекции:")
    
    for name, correction_func in corrections:
        corrected = correction_func(test_depth)
        
        print(f"\n   {name}:")
        print(f"      Диапазон: {corrected.min()}мм - {corrected.max()}мм")
        print(f"      Среднее: {corrected.mean():.1f}мм")
        
        # Вычисляем статистику изменений
        if name != "Исходное изображение":
            diff = corrected.astype(np.float32) - test_depth.astype(np.float32)
            print(f"      Изменение: {diff.mean():.1f}±{diff.std():.1f}мм")
    
    print("\n🎯 Рекомендации по настройке параметров:")
    print("   • Depth Offset: начните с +20мм...+50мм для отдаления виртуальной камеры")
    print("   • Depth Scale: используйте 1.05x...1.1x для небольшого масштабирования")
    print("   • Spatial Filter: включите с ядром 5x5 для уменьшения шума")
    print("   • Touch Threshold: увеличьте на 50-100% при активной коррекции")
    
    print("\n⚡ Управление в реальном времени:")
    print("   • Все параметры доступны через ползунки в окне управления")
    print("   • Изменения применяются мгновенно без перезапуска")
    print("   • Комбинируйте методы для оптимального результата")
    
    print("\n✅ Демонстрация завершена!")

if __name__ == "__main__":
    demonstrate_correction_methods()
