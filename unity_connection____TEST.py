import socket
import json
import time

class InteractiveUnityClient:
    def __init__(self, host='127.0.0.1', port=8052):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Интерактивный клиент запущен. Отправка на {host}:{port}")
        print("Команды:")
        print("  x,y - отправить касание (например: 0.5,0.3)")
        print("  multi - отправить несколько касаний")
        print("  exit - выход")
    
    def send_custom_data(self, x, y, touch_type="touch"):
        """Отправка кастомных данных"""
        message = {
            "touches": [{"x": x, "y": y, "type": touch_type}],
            "timestamp": time.time()
        }
        
        try:
            json_data = json.dumps(message)
            self.sock.sendto(json_data.encode(), (self.host, self.port))
            print(f"Отправлено: X={x}, Y={y}, Type={touch_type}")
            return True
        except Exception as e:
            print(f"Ошибка отправки: {e}")
            return False
    
    def run_interactive(self):
        """Интерактивный режим"""
        while True:
            try:
                command = input("Введите команду: ").strip().lower()
                
                if command == "exit":
                    break
                elif command == "multi":
                    # Отправляем несколько касаний
                    for i in range(3):
                        x = 0.2 + i * 0.3
                        y = 0.2 + i * 0.1
                        self.send_custom_data(x, y, "touch" if i % 2 == 0 else "drag")
                        time.sleep(0.5)
                elif "," in command:
                    # Парсим координаты
                    parts = command.split(",")
                    if len(parts) == 2:
                        try:
                            x = float(parts[0].strip())
                            y = float(parts[1].strip())
                            if 0 <= x <= 1 and 0 <= y <= 1:
                                self.send_custom_data(x, y)
                            else:
                                print("Координаты должны быть между 0 и 1")
                        except ValueError:
                            print("Неверный формат координат")
                    else:
                        print("Используйте формат: x,y")
                else:
                    print("Неизвестная команда")
            except KeyboardInterrupt:
                print("\nВыход...")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
    
    def close(self):
        """Закрытие соединения"""
        self.sock.close()

# Запуск интерактивного клиента
if __name__ == "__main__":
    client = InteractiveUnityClient()
    client.run_interactive()
    client.close()