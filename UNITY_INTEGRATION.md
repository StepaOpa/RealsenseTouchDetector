# Unity Integration Guide
## RealSense Touch Detector to Unity Communication

### 🎯 Обзор

Система передает координаты касаний из RealSense Touch Detector в Unity в реальном времени через UDP протокол.

### 📡 Протокол коммуникации

**Порт:** 8052 (UDP)  
**Формат:** JSON  
**Кодировка:** UTF-8

### 📋 Структура данных

#### Touch Data Message
```json
{
  "type": "touch_data",
  "timestamp": 1699123456.789,
  "touches": [
    {
      "id": 0,
      "x": 0.5,        // Нормализованная X (0.0-1.0)
      "y": 0.3,        // Нормализованная Y (0.0-1.0)
      "depth": 0.125,  // Глубина в метрах
      "area": 1500,    // Площадь в пикселях
      "confidence": 0.95, // Уверенность (0.0-1.0)
      "timestamp": 1699123456.789,
      "type": "touch"
    }
  ],
  "touch_count": 1,
  "frame_id": 123
}
```

#### Connection Test Message
```json
{
  "type": "connection_test",
  "timestamp": 1699123456.789,
  "message": "Test connection from RealSense Touch Detector"
}
```

#### System Status Message  
```json
{
  "type": "system_status",
  "timestamp": 1699123456.789,
  "status": {
    "camera_connected": true,
    "touch_detection_enabled": true,
    "total_touches_detected": 1250
  }
}
```

### 🎮 Unity Integration

#### 1. Добавьте TouchReceiver.cs в проект

1. Скопируйте файл `unity_touch_receiver.cs` в ваш Unity проект
2. Добавьте скрипт на любой GameObject в сцене
3. Настройте параметры в инспекторе:
   - **Listen Port**: 8052 (по умолчанию)
   - **Touch Prefab**: префаб для визуализации касаний
   - **Touch Parent**: родительский объект для касаний
   - **Touch Lifetime**: время жизни касания (0.5 сек)

#### 2. Зависимости

Убедитесь, что у вас установлен **Newtonsoft.Json**:
- Window → Package Manager → Unity Registry
- Найдите "com.unity.nuget.newtonsoft-json"
- Нажмите Install

#### 3. Настройка координат

По умолчанию TouchReceiver конвертирует нормализованные координаты (0.0-1.0) в мировые координаты Unity. Измените метод `NormalizedToWorldPosition()` под вашу сцену:

```csharp
private Vector3 NormalizedToWorldPosition(float normalizedX, float normalizedY)
{
    // Настройте эти значения под вашу сцену
    float worldX = (normalizedX - 0.5f) * 10f; // От -5 до +5
    float worldZ = (normalizedY - 0.5f) * 10f; // От -5 до +5
    float worldY = 0f; // Высота касания
    
    return new Vector3(worldX, worldY, worldZ);
}
```

### ⚙️ Настройка RealSense Touch Detector

В окне управления RealSense Touch Detector:

1. **Unity отправка**: включите ползунок (ON)
2. **Unity порт**: установите 8052 (или другой порт)
3. **IP адрес**: по умолчанию 127.0.0.1 (localhost)

### 🔧 Параметры в UI

| Параметр | Диапазон | Описание |
|----------|----------|----------|
| Unity отправка | ON/OFF | Включение/выключение передачи |
| Unity порт | 8000-9000 | UDP порт для отправки |

### 🧪 Тестирование без Unity

Для тестирования без Unity используйте `test_unity_receiver.py`:

```bash
python test_unity_receiver.py
```

Этот скрипт:
- ✅ Прослушивает порт 8052
- ✅ Отображает полученные касания в консоли
- ✅ Показывает статистику передачи
- ✅ Поддерживает интерактивные команды

### 📊 Мониторинг

#### В Unity (TouchReceiver GUI):
- **UDP Status**: состояние прослушивания
- **Active Touches**: количество активных касаний
- **Total Received**: всего получено касаний
- **Connection**: статус соединения

#### В RealSense Touch Detector:
- **Unity**: статус (ON/OFF) и адрес
- **Sent**: количество отправленных пакетов
- **Errors**: количество ошибок отправки

### 🚀 Производительность

- **Частота отправки**: до 1000 пакетов/сек
- **Задержка**: < 1мс на локальной машине
- **Максимум касаний**: 10 касаний на кадр
- **Размер пакета**: ~300-500 байт

### 🔍 Отладка

#### Проблемы подключения:
1. Проверьте, что Unity слушает правильный порт
2. Убедитесь, что firewall не блокирует UDP трафик
3. Проверьте IP адрес (127.0.0.1 для localhost)

#### Потеря пакетов:
1. Увеличьте буфер UDP сокета в Unity
2. Уменьшите частоту отправки в RealSense
3. Проверьте производительность системы

#### Задержка касаний:
1. Используйте проводное подключение
2. Закройте ненужные приложения
3. Оптимизируйте обработку в Unity

### 📝 Пример использования в Unity

```csharp
public class TouchHandler : MonoBehaviour
{
    private TouchReceiver touchReceiver;
    
    void Start()
    {
        touchReceiver = FindObjectOfType<TouchReceiver>();
    }
    
    void Update()
    {
        // Получаем количество активных касаний
        int activeTouches = touchReceiver.GetActiveTouchCount();
        
        if (activeTouches > 0)
        {
            Debug.Log($"Активных касаний: {activeTouches}");
        }
    }
}
```

### 🎯 Следующие шаги

1. **Калибровка координат**: настройте преобразование под вашу проекционную поверхность
2. **Обработка жестов**: добавьте распознавание жестов (свайп, тап, пинч)
3. **Многопользовательский режим**: поддержка нескольких источников касаний
4. **Буферизация**: добавьте буферизацию для сглаживания движений

### 📞 Поддержка

При возникновении проблем:
1. Проверьте логи Unity Console
2. Используйте `test_unity_receiver.py` для диагностики
3. Проверьте настройки firewall/antivirus
4. Убедитесь в корректности IP адреса и порта
