"""
Главное приложение для системы отслеживания ног с камерой Intel RealSense D435f

Автор: AI Assistant
Версия: 1.0
"""

import cv2
import sys
import argparse
from typing import Optional

# Импорт модулей системы
from realsense_camera import RealSenseManager
from foot_tracker import FootTracker
from visualization import VisualizationManager
from calibration import InteractiveCalibrator
from config import ConfigManager


class FootTrackingApp:
    """Главное приложение для отслеживания ног"""
    
    def __init__(self) -> None:
        self.tracker: Optional[FootTracker] = None
        self.visualizer: Optional[VisualizationManager] = None
        self.running = False
        
    def initialize(self) -> bool:
        """Инициализация системы"""
        print("=== Система отслеживания ног RealSense D435f ===")
        print("Инициализация...")
        
        try:
            # Проверяем доступность библиотек
            print("Проверка библиотек...")
            import numpy as np
            import cv2
            import pyrealsense2 as rs
            print("✅ Все библиотеки доступны")
            
            print("Инициализация трекера...")
            self.tracker = FootTracker()
            if not self.tracker.initialize():
                print("❌ Не удалось инициализировать трекер")
                return False
                
            print("Инициализация визуализатора...")
            self.visualizer = VisualizationManager()
            print("✅ Система готова к работе!")
            return True
            
        except ImportError as e:
            print(f"❌ Ошибка импорта библиотеки: {e}")
            print("Установите зависимости: pip install -r requirements.txt")
            return False
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_tracking(self) -> None:
        """Запуск основного режима отслеживания"""
        print("\nЗапуск отслеживания ног...")
        self._print_controls()
        
        cv2.namedWindow('Camera View')
        cv2.namedWindow('Screen Space') 
        cv2.namedWindow('Floor Mask')
        
        self.running = True
        
        try:
            while self.running:
                try:
                    # Получение данных от трекера
                    debug_image, mask_image, foot_positions = self.tracker.process_frame()
                    
                    if debug_image is None:
                        continue
                    
                    # Создание визуализаций
                    camera_view = self.visualizer.create_camera_view(
                        debug_image, mask_image, [], foot_positions,
                        calibration_points=self.tracker.calibration.calibration_points,
                        is_calibrated=self.tracker.calibration.is_calibrated
                    )
                    screen_view = self.visualizer.create_screen_view(foot_positions)
                    
                    # Отображение (с проверкой существования изображений)
                    if camera_view is not None:
                        cv2.imshow('Camera View', camera_view)
                    if screen_view is not None:
                        cv2.imshow('Screen Space', screen_view)
                    if mask_image is not None:
                        cv2.imshow('Floor Mask', mask_image)
                    
                    # Вывод координат в консоль
                    if foot_positions:
                        positions_str = ", ".join([
                            f"Стопа-{foot.id}:cam({foot.camera_x},{foot.camera_y})->screen({foot.screen_x},{foot.screen_y})"
                            for foot in foot_positions
                        ])
                        print(f"\rСтоп: {len(foot_positions)} | {positions_str}", end="")
                        
                        # Дополнительная отладка для первой стопы
                        if len(foot_positions) > 0:
                            first_foot = foot_positions[0]
                            if self.tracker.calibration.is_calibrated:
                                print(f"\n🔍 ОТЛАДКА стопы {first_foot.id}:")
                                print(f"   Камера: ({first_foot.camera_x}, {first_foot.camera_y})")
                                print(f"   Экран:  ({first_foot.screen_x}, {first_foot.screen_y})")
                                
                                # Проверим преобразование заново с отладкой
                                debug_x, debug_y = self.tracker.calibration.transform_point(
                                    first_foot.camera_x, first_foot.camera_y, debug=True
                                )
                    
                    # Обработка клавиш
                    if not self._handle_keys():
                        break
                        
                except Exception as frame_error:
                    print(f"\nОшибка обработки кадра: {frame_error}")
                    # Продолжаем работу, не выходим из-за одной ошибки
                    continue
                    
        except KeyboardInterrupt:
            print("\nОстановка программы...")
        finally:
            self.running = False
            cv2.destroyAllWindows()
    
    def run_calibration(self) -> None:
        """Запуск режима калибровки проекции"""
        print("\n=== КАЛИБРОВКА ПРОЕКЦИИ ===")
        print("Кликните мышью по 4 углам проекционной области")
        print("ESC - выход, R - сброс, S - сохранить")
        
        calibrator = InteractiveCalibrator(self.tracker.calibration)
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', calibrator.mouse_callback)
        
        self.running = True
        
        try:
            while self.running:
                try:
                    depth_image, color_image = self.tracker.camera.get_frames()
                    if color_image is None:
                        continue
                    
                    overlay_image = calibrator.run_calibration(color_image)
                    if overlay_image is not None:
                        cv2.imshow('Calibration', overlay_image)
                        
                except Exception as cal_error:
                    print(f"\nОшибка в калибровке: {cal_error}")
                    continue
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r') or key == ord('R'):
                    self.tracker.calibration.reset_calibration()
                elif key == ord('s') or key == ord('S'):
                    self.tracker.calibration.save_calibration()
                    
        except KeyboardInterrupt:
            print("\nОстановка калибровки...")
        finally:
            cv2.destroyAllWindows()
    
    def _handle_keys(self) -> bool:
        """Обработка нажатий клавиш"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - выход
            return False
        elif key == ord('c') or key == ord('C'):  # Автокалибровка пола
            self.tracker.calibrate_floor_depth()
        elif key == ord('+') or key == ord('='):  # Увеличить глубину пола
            new_depth = self.tracker.floor_detector.floor_depth + 0.05
            self.tracker.set_floor_depth(new_depth)
            print(f"\nГлубина пола: {new_depth:.2f}м")
        elif key == ord('-'):  # Уменьшить глубину пола
            new_depth = max(0.5, self.tracker.floor_detector.floor_depth - 0.05)
            self.tracker.set_floor_depth(new_depth)
            print(f"\nГлубина пола: {new_depth:.2f}м")
        elif key == ord('s') or key == ord('S'):  # Сохранить настройки
            self.tracker.calibration.save_calibration()
            print("\nНастройки сохранены")
        elif key == ord('t') or key == ord('T'):  # Переключить следы
            self.visualizer.toggle_trails()
        elif key == ord('g') or key == ord('G'):  # Переключить сетку
            self.visualizer.toggle_grid()
        elif key == ord('r') or key == ord('R'):  # Очистить следы
            self.visualizer.clear_trails()
        elif key == ord('d') or key == ord('D'):  # Отладочная информация
            self._show_debug_info()
        elif key == ord('x') or key == ord('X'):  # Сброс калибровки
            if self.tracker and self.tracker.calibration:
                self.tracker.calibration.reset_calibration()
                print("\n✅ Калибровка сброшена! Нажмите ESC для перехода к новой калибровке.")
        elif key == ord('m') or key == ord('M'):  # Тест преобразования координат
            self._test_coordinate_mapping()
        elif key == ord('f') or key == ord('F'):  # Принудительный пересчет калибровки
            self._force_recalculate_calibration()
        
        return True
    
    def _show_debug_info(self) -> None:
        """Показать отладочную информацию о калибровке"""
        print("\n" + "="*50)
        print("ОТЛАДОЧНАЯ ИНФОРМАЦИЯ")
        print("="*50)
        
        if self.tracker and self.tracker.calibration:
            cal = self.tracker.calibration
            print(f"Статус калибровки: {'✅ Готова' if cal.is_calibrated else '❌ Не готова'}")
            print(f"Разрешение камеры: {cal.camera_width}x{cal.camera_height}")
            print(f"Целевое разрешение: {cal.target_width}x{cal.target_height}")
            print(f"Точек калибровки: {len(cal.calibration_points)}/4")
            
            if cal.calibration_points:
                print("Точки калибровки:")
                for i, (x, y) in enumerate(cal.calibration_points):
                    print(f"  {i+1}. ({x}, {y})")
            
            # Тест нескольких точек
            if cal.is_calibrated:
                print("\nТест преобразования координат:")
                test_coords = [(100, 100), (320, 240), (540, 380)]
                for cx, cy in test_coords:
                    sx, sy = cal.transform_point(cx, cy, debug=True)
                    
        print("="*50)
    
    def _test_coordinate_mapping(self) -> None:
        """Тестирование преобразования координат в различных точках"""
        print("\n" + "="*60)
        print("ТЕСТ ПРЕОБРАЗОВАНИЯ КООРДИНАТ")
        print("="*60)
        
        if not self.tracker or not self.tracker.calibration.is_calibrated:
            print("❌ Калибровка не готова!")
            return
        
        cal = self.tracker.calibration
        print(f"Разрешение камеры: {cal.camera_width}x{cal.camera_height}")
        print(f"Целевое разрешение: {cal.target_width}x{cal.target_height}")
        print(f"Точки калибровки: {len(cal.calibration_points)}")
        
        if len(cal.calibration_points) != 4:
            print("❌ Недостаточно точек калибровки!")
            return
        
        print("\nТестовые точки:")
        
        # Тест углов области калибровки
        corners = cal.calibration_points
        corner_names = ["TL", "TR", "BR", "BL"]
        expected_corners = [
            (0, 0), 
            (cal.target_width-1, 0), 
            (cal.target_width-1, cal.target_height-1), 
            (0, cal.target_height-1)
        ]
        
        for i, ((cx, cy), name, (exp_x, exp_y)) in enumerate(zip(corners, corner_names, expected_corners)):
            act_x, act_y = cal.transform_point(cx, cy, debug=False)
            error_x = abs(act_x - exp_x)
            error_y = abs(act_y - exp_y)
            status = "✅" if (error_x < 50 and error_y < 50) else "❌"
            print(f"  {status} {name}: камера({cx},{cy}) -> ожидается({exp_x},{exp_y}) -> получили({act_x},{act_y}) [ошибка: {error_x},{error_y}]")
        
        # Тест центра области
        center_x = sum(p[0] for p in corners) // 4
        center_y = sum(p[1] for p in corners) // 4
        center_screen_x, center_screen_y = cal.transform_point(center_x, center_y, debug=True)
        expected_center_x = cal.target_width // 2
        expected_center_y = cal.target_height // 2
        
        print(f"\n🎯 ЦЕНТР области:")
        print(f"   Камера: ({center_x}, {center_y})")
        print(f"   Ожидается: ({expected_center_x}, {expected_center_y})")
        print(f"   Получили: ({center_screen_x}, {center_screen_y})")
        
        # Дополнительные тестовые точки
        test_points = [
            ("Четверть X, четверть Y", center_x//2, center_y//2),
            ("Три четверти X, четверть Y", center_x + center_x//2, center_y//2),
            ("Четверть X, три четверти Y", center_x//2, center_y + center_y//2),
            ("Три четверти X, три четверти Y", center_x + center_x//2, center_y + center_y//2),
        ]
        
        print(f"\n🧪 ДОПОЛНИТЕЛЬНЫЕ ТОЧКИ:")
        for name, tx, ty in test_points:
            screen_x, screen_y = cal.transform_point(tx, ty, debug=False)
            print(f"   {name}: камера({tx},{ty}) -> экран({screen_x},{screen_y})")
        
        print("="*60)
    
    def _force_recalculate_calibration(self) -> None:
        """Принудительный пересчет калибровки с улучшенным алгоритмом"""
        print("\n" + "="*50)
        print("ПРИНУДИТЕЛЬНЫЙ ПЕРЕСЧЕТ КАЛИБРОВКИ")
        print("="*50)
        
        if not self.tracker or not self.tracker.calibration:
            print("❌ Система не инициализирована!")
            return
        
        cal = self.tracker.calibration
        if len(cal.calibration_points) != 4:
            print("❌ Недостаточно точек калибровки! Выполните калибровку сначала.")
            return
        
        print("Текущие точки калибровки:")
        for i, (x, y) in enumerate(cal.calibration_points):
            print(f"  Точка {i+1}: ({x}, {y})")
        
        print("\n🔄 Пересчитываем гомографию с улучшенным алгоритмом...")
        
        # Сохраняем текущее состояние
        old_calibrated = cal.is_calibrated
        
        # Принудительно запускаем пересчет
        cal._calculate_homography()
        
        if cal.is_calibrated:
            print("✅ Калибровка успешно пересчитана!")
            
            # Тестируем результат
            center_x = sum(p[0] for p in cal.calibration_points) // 4
            center_y = sum(p[1] for p in cal.calibration_points) // 4
            screen_x, screen_y = cal.transform_point(center_x, center_y)
            expected_x = cal.target_width // 2
            expected_y = cal.target_height // 2
            
            print(f"🎯 Тест центра области:")
            print(f"   Камера: ({center_x}, {center_y})")
            print(f"   Экран:  ({screen_x}, {screen_y})")
            print(f"   Ожидается: ({expected_x}, {expected_y})")
            
            error_x = abs(screen_x - expected_x)
            error_y = abs(screen_y - expected_y)
            if error_x < 100 and error_y < 100:
                print("✅ Результат выглядит корректно!")
            else:
                print("⚠️ Большая ошибка! Возможно, нужна новая калибровка.")
                
        else:
            print("❌ Ошибка пересчета калибровки!")
        
        print("="*50)
    
    def _quick_recalibrate(self) -> None:
        """Быстрая рекалибровка - сброс и новая калибровка"""
        print("\n=== БЫСТРАЯ РЕКАЛИБРОВКА ===")
        
        if self.tracker and self.tracker.calibration:
            # Сброс текущей калибровки
            self.tracker.calibration.reset_calibration()
            print("✅ Текущая калибровка сброшена")
            
            print("\nТеперь выполним новую калибровку...")
            print("ВАЖНО: Кликайте по углам в любом порядке - система автоматически определит правильный порядок!")
            
            # Переходим к калибровке
            self.run_calibration()
        else:
            print("❌ Ошибка: система не инициализирована")
    
    def _print_controls(self) -> None:
        """Вывод управления"""
        print("\nУправление:")
        print("  ESC - выход")
        print("  C   - автокалибровка пола")
        print("  +/- - изменить глубину пола")
        print("  S   - сохранить настройки")
        print("  T   - переключить следы")
        print("  G   - переключить сетку")
        print("  R   - очистить следы")
        print("  D   - отладочная информация")
        print("  M   - тест преобразования координат")
        print("  F   - пересчитать калибровку (улучшенный алгоритм)")
        print("  X   - сбросить калибровку проекции")
    
    def show_menu(self) -> None:
        """Отображение меню"""
        while True:
            print("\n" + "="*50)
            print("  СИСТЕМА ОТСЛЕЖИВАНИЯ НОГ REALSENSE D435F")
            print("="*50)
            print("1. Отслеживание ног (основной режим)")
            print("2. Калибровка проекции")
            print("3. Быстрая рекалибровка (сброс + новая калибровка)")
            print("0. Выход")
            print("-"*50)
            
            choice = input("Выберите режим (0-3): ").strip()
            
            if choice == "1":
                self.run_tracking()
            elif choice == "2":
                self.run_calibration()
            elif choice == "3":
                self._quick_recalibrate()
            elif choice == "0":
                break
            else:
                print("Неверный выбор!")
    
    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.tracker:
            self.tracker.cleanup()
        cv2.destroyAllWindows()


def main() -> None:
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Система отслеживания ног RealSense")
    parser.add_argument("--mode", choices=["tracking", "calibration"], 
                       help="Прямой запуск режима")
    
    args = parser.parse_args()
    
    try:
        app = FootTrackingApp()
        
        if not app.initialize():
            print("Ошибка инициализации. Проверьте подключение камеры.")
            sys.exit(1)
        
        if args.mode == "tracking":
            app.run_tracking()
        elif args.mode == "calibration":
            app.run_calibration()
        else:
            app.show_menu()
            
    except Exception as e:
        print(f"Критическая ошибка: {e}")
    finally:
        if 'app' in locals():
            app.cleanup()


if __name__ == "__main__":
    main()