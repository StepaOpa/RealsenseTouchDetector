#!/usr/bin/env python3
"""
Тестовый UDP сервер для имитации Unity приемника
Получает данные касаний и отображает их в консоли
"""

import socket
import json
import time
import threading
from typing import Dict, Any


class TestUnityReceiver:
    """Тестовый приемник Unity данных"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8052) -> None:
        self.host = host
        self.port = port
        self.sock = None
        self.is_running = False
        self.receive_thread = None
        
        # Статистика
        self.total_received = 0
        self.total_touches = 0
        self.last_frame_id = 0
        
    def start(self) -> None:
        """Запуск тестового сервера"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(1.0)
            self.is_running = True
            
            print(f"🎮 Тестовый Unity приемник запущен на {self.host}:{self.port}")
            print("📡 Ожидание данных касаний...")
            print("-" * 60)
            
            # Запускаем поток приема данных
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
        except Exception as e:
            print(f"❌ Ошибка запуска сервера: {e}")
    
    def stop(self) -> None:
        """Остановка сервера"""
        self.is_running = False
        if self.sock:
            self.sock.close()
        print("🛑 Тестовый Unity приемник остановлен")
    
    def _receive_loop(self) -> None:
        """Основной цикл приема данных"""
        while self.is_running:
            try:
                # Получаем данные
                data, address = self.sock.recvfrom(4096)
                message = data.decode('utf-8')
                
                # Парсим JSON
                try:
                    json_data = json.loads(message)
                    self._process_message(json_data, address)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга JSON: {e}")
                    
            except socket.timeout:
                continue  # Таймаут - нормально, продолжаем
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Ошибка приема данных: {e}")
    
    def _process_message(self, data: Dict[str, Any], address: tuple) -> None:
        """Обработка полученного сообщения"""
        msg_type = data.get('type', 'unknown')
        timestamp = data.get('timestamp', 0)
        
        if msg_type == 'touch_data':
            self._process_touch_data(data, address)
        elif msg_type == 'connection_test':
            self._process_connection_test(data, address)
        elif msg_type == 'system_status':
            self._process_system_status(data, address)
        else:
            print(f"🔍 Неизвестный тип сообщения: {msg_type}")
    
    def _process_touch_data(self, data: Dict[str, Any], address: tuple) -> None:
        """Обработка данных касаний"""
        self.total_received += 1
        
        touches = data.get('touches', [])
        touch_count = data.get('touch_count', 0)
        frame_id = data.get('frame_id', 0)
        timestamp = data.get('timestamp', 0)
        
        self.total_touches += touch_count
        self.last_frame_id = frame_id
        
        # Выводим информацию о касаниях
        current_time = time.strftime("%H:%M:%S")
        print(f"📱 [{current_time}] Frame #{frame_id}: {touch_count} касаний от {address[0]}")
        
        for i, touch in enumerate(touches):
            touch_id = touch.get('id', i)
            x = touch.get('x', 0)
            y = touch.get('y', 0)
            depth = touch.get('depth', 0)
            area = touch.get('area', 0)
            confidence = touch.get('confidence', 0)
            
            # Конвертируем нормализованные координаты в пиксели (для наглядности)
            screen_x = int(x * 1920)
            screen_y = int(y * 1080)
            
            print(f"   👆 Touch {touch_id}: ({x:.3f}, {y:.3f}) -> ({screen_x}, {screen_y})")
            print(f"      💪 Confidence: {confidence:.3f}, Area: {area}px, Depth: {depth:.3f}m")
        
        # Выводим статистику каждые 10 кадров
        if frame_id % 10 == 0:
            print(f"📊 Статистика: {self.total_received} пакетов, {self.total_touches} касаний")
            print("-" * 40)
    
    def _process_connection_test(self, data: Dict[str, Any], address: tuple) -> None:
        """Обработка тестового соединения"""
        message = data.get('message', 'No message')
        timestamp = data.get('timestamp', 0)
        
        print(f"🔌 Тест соединения от {address[0]}: {message}")
        print(f"   ⏰ Время: {time.ctime(timestamp)}")
    
    def _process_system_status(self, data: Dict[str, Any], address: tuple) -> None:
        """Обработка системного статуса"""
        status = data.get('status', {})
        
        print(f"🖥️ Системный статус от {address[0]}:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    def print_statistics(self) -> None:
        """Вывод подробной статистики"""
        print("\n" + "=" * 50)
        print("📊 СТАТИСТИКА ТЕСТОВОГО UNITY ПРИЕМНИКА")
        print("=" * 50)
        print(f"📡 Адрес сервера: {self.host}:{self.port}")
        print(f"🔄 Статус: {'🟢 Работает' if self.is_running else '🔴 Остановлен'}")
        print(f"📦 Всего пакетов получено: {self.total_received}")
        print(f"👆 Всего касаний обработано: {self.total_touches}")
        print(f"🎬 Последний Frame ID: {self.last_frame_id}")
        
        if self.total_received > 0:
            avg_touches = self.total_touches / self.total_received
            print(f"📈 Среднее касаний на кадр: {avg_touches:.2f}")
        
        print("=" * 50)


def main():
    """Основная функция для запуска тестового приемника"""
    print("🎮 Запуск тестового Unity приемника касаний")
    print("💡 Этот скрипт имитирует Unity приложение для тестирования")
    print("🔌 Подключите ваш RealSense Touch Detector к порту 8052")
    print()
    
    receiver = TestUnityReceiver()
    
    try:
        receiver.start()
        
        print("⌨️ Команды:")
        print("   's' - показать статистику")
        print("   'q' - выход")
        print()
        
        # Основной цикл команд
        while receiver.is_running:
            try:
                command = input().strip().lower()
                
                if command == 'q':
                    break
                elif command == 's':
                    receiver.print_statistics()
                elif command == '':
                    continue  # Игнорируем пустой ввод
                else:
                    print("❓ Неизвестная команда. Используйте 's' или 'q'")
                    
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        print("\n🔸 Прерывание от пользователя")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    finally:
        receiver.stop()
        receiver.print_statistics()


if __name__ == "__main__":
    main()
