using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

/// <summary>
/// Unity скрипт для приема данных касаний от RealSense Touch Detector
/// Поместите этот скрипт на любой GameObject в сцене
/// </summary>
public class TouchReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    [SerializeField] private int listenPort = 8052;
    [SerializeField] private bool autoStart = true;

    [Header("Touch Visualization")]
    [SerializeField] private GameObject touchPrefab;  // Префаб для отображения касаний
    [SerializeField] private Transform touchParent;   // Родительский объект для касаний
    [SerializeField] private float touchLifetime = 0.5f; // Время жизни касания

    [Header("Debug")]
    [SerializeField] private bool showDebugInfo = true;
    [SerializeField] private bool logTouchData = false;

    // UDP Socket
    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private bool isListening = false;

    // Touch data
    private Dictionary<int, TouchObject> activeTouches = new Dictionary<int, TouchObject>();
    private int totalTouchesReceived = 0;
    private float lastReceiveTime = 0f;

    // UI для отображения статистики
    private GUIStyle guiStyle;

    [System.Serializable]
    public class TouchData
    {
        public string type;
        public float timestamp;
        public TouchPoint[] touches;
        public int touch_count;
        public int frame_id;
    }

    [System.Serializable]
    public class TouchPoint
    {
        public int id;
        public float x;      // Нормализованная координата X (0.0-1.0)
        public float y;      // Нормализованная координата Y (0.0-1.0)
        public float depth;  // Глубина в метрах
        public int area;     // Площадь в пикселях
        public float confidence; // Уверенность (0.0-1.0)
        public float timestamp;
        public string type;
    }

    private class TouchObject
    {
        public GameObject gameObject;
        public float creationTime;
        public TouchPoint lastData;

        public TouchObject(GameObject go, TouchPoint data)
        {
            gameObject = go;
            creationTime = Time.time;
            lastData = data;
        }
    }

    void Start()
    {
        // Настройка GUI стиля
        guiStyle = new GUIStyle();
        guiStyle.fontSize = 16;
        guiStyle.normal.textColor = Color.white;

        // Автозапуск если включен
        if (autoStart)
        {
            StartListening();
        }

        Debug.Log($"TouchReceiver инициализирован. Порт: {listenPort}");
    }

    public void StartListening()
    {
        try
        {
            // Создаем UDP клиент
            udpClient = new UdpClient(listenPort);
            remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
            isListening = true;

            Debug.Log($"Начато прослушивание UDP порта {listenPort}");

            // Начинаем асинхронное получение данных
            udpClient.BeginReceive(OnDataReceived, null);
        }
        catch (Exception e)
        {
            Debug.LogError($"Ошибка запуска UDP клиента: {e.Message}");
        }
    }

    public void StopListening()
    {
        isListening = false;

        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }

        // Очищаем все активные касания
        ClearAllTouches();

        Debug.Log("UDP прослушивание остановлено");
    }

    private void OnDataReceived(IAsyncResult result)
    {
        if (!isListening || udpClient == null) return;

        try
        {
            // Получаем данные
            byte[] data = udpClient.EndReceive(result, ref remoteEndPoint);
            string jsonString = Encoding.UTF8.GetString(data);

            // Парсим JSON
            TouchData touchData = JsonConvert.DeserializeObject<TouchData>(jsonString);

            if (touchData != null && touchData.type == "touch_data")
            {
                ProcessTouchData(touchData);
            }

            // Продолжаем слушать
            udpClient.BeginReceive(OnDataReceived, null);
        }
        catch (Exception e)
        {
            if (isListening)
            {
                Debug.LogError($"Ошибка получения данных: {e.Message}");
                // Продолжаем слушать даже после ошибки
                if (udpClient != null)
                {
                    udpClient.BeginReceive(OnDataReceived, null);
                }
            }
        }
    }

    private void ProcessTouchData(TouchData data)
    {
        lastReceiveTime = Time.time;
        totalTouchesReceived += data.touch_count;

        if (logTouchData)
        {
            Debug.Log($"Получено {data.touch_count} касаний, Frame ID: {data.frame_id}");
        }

        // Обрабатываем каждое касание
        foreach (TouchPoint touch in data.touches)
        {
            ProcessTouch(touch);
        }

        // Удаляем старые касания
        RemoveOldTouches();
    }

    private void ProcessTouch(TouchPoint touch)
    {
        // Конвертируем нормализованные координаты в мировые координаты Unity
        Vector3 worldPosition = NormalizedToWorldPosition(touch.x, touch.y);

        if (activeTouches.ContainsKey(touch.id))
        {
            // Обновляем существующее касание
            TouchObject existingTouch = activeTouches[touch.id];
            existingTouch.gameObject.transform.position = worldPosition;
            existingTouch.lastData = touch;
            existingTouch.creationTime = Time.time; // Обновляем время для продления жизни
        }
        else
        {
            // Создаем новое касание
            GameObject touchGO = CreateTouchObject(worldPosition, touch);
            activeTouches[touch.id] = new TouchObject(touchGO, touch);
        }

        if (logTouchData)
        {
            Debug.Log($"Touch {touch.id}: ({touch.x:F3}, {touch.y:F3}) -> World: {worldPosition}, Confidence: {touch.confidence:F3}");
        }
    }

    private Vector3 NormalizedToWorldPosition(float normalizedX, float normalizedY)
    {
        // Простое преобразование в мировые координаты
        // Настройте эти значения под вашу сцену
        float worldX = (normalizedX - 0.5f) * 10f; // Диапазон от -5 до +5
        float worldZ = (normalizedY - 0.5f) * 10f; // Диапазон от -5 до +5
        float worldY = 0f; // Высота касания

        return new Vector3(worldX, worldY, worldZ);
    }

    private GameObject CreateTouchObject(Vector3 position, TouchPoint touchData)
    {
        GameObject touchGO;

        if (touchPrefab != null)
        {
            touchGO = Instantiate(touchPrefab, position, Quaternion.identity);
        }
        else
        {
            // Создаем простую сферу если нет префаба
            touchGO = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            touchGO.transform.localScale = Vector3.one * 0.1f;

            // Добавляем цвет в зависимости от уверенности
            Renderer renderer = touchGO.GetComponent<Renderer>();
            if (renderer != null)
            {
                Color color = Color.Lerp(Color.red, Color.green, touchData.confidence);
                renderer.material.color = color;
            }
        }

        touchGO.transform.position = position;
        touchGO.name = $"Touch_{touchData.id}";

        if (touchParent != null)
        {
            touchGO.transform.SetParent(touchParent);
        }

        return touchGO;
    }

    private void RemoveOldTouches()
    {
        List<int> touchesToRemove = new List<int>();

        foreach (var kvp in activeTouches)
        {
            if (Time.time - kvp.Value.creationTime > touchLifetime)
            {
                touchesToRemove.Add(kvp.Key);
            }
        }

        foreach (int touchId in touchesToRemove)
        {
            if (activeTouches[touchId].gameObject != null)
            {
                DestroyImmediate(activeTouches[touchId].gameObject);
            }
            activeTouches.Remove(touchId);
        }
    }

    private void ClearAllTouches()
    {
        foreach (var kvp in activeTouches)
        {
            if (kvp.Value.gameObject != null)
            {
                DestroyImmediate(kvp.Value.gameObject);
            }
        }
        activeTouches.Clear();
    }

    void Update()
    {
        // Автоматически удаляем старые касания
        RemoveOldTouches();
    }

    void OnGUI()
    {
        if (!showDebugInfo) return;

        // Отображаем статистику
        string status = isListening ? "LISTENING" : "STOPPED";
        Color statusColor = isListening ? Color.green : Color.red;

        GUI.color = statusColor;
        GUI.Label(new Rect(10, 10, 300, 30), $"UDP Status: {status} (Port: {listenPort})", guiStyle);

        GUI.color = Color.white;
        GUI.Label(new Rect(10, 35, 300, 25), $"Active Touches: {activeTouches.Count}", guiStyle);
        GUI.Label(new Rect(10, 60, 300, 25), $"Total Received: {totalTouchesReceived}", guiStyle);

        float timeSinceLastReceive = Time.time - lastReceiveTime;
        string connectionStatus = timeSinceLastReceive < 2f ? "CONNECTED" : "NO DATA";
        Color connectionColor = timeSinceLastReceive < 2f ? Color.green : Color.yellow;

        GUI.color = connectionColor;
        GUI.Label(new Rect(10, 85, 300, 25), $"Connection: {connectionStatus}", guiStyle);

        // Кнопки управления
        GUI.color = Color.white;
        if (GUI.Button(new Rect(10, 120, 100, 30), isListening ? "Stop" : "Start"))
        {
            if (isListening)
                StopListening();
            else
                StartListening();
        }

        if (GUI.Button(new Rect(120, 120, 100, 30), "Clear Touches"))
        {
            ClearAllTouches();
        }
    }

    void OnDestroy()
    {
        StopListening();
    }

    void OnApplicationQuit()
    {
        StopListening();
    }

    // Публичные методы для внешнего управления
    public int GetActiveTouchCount() => activeTouches.Count;
    public int GetTotalTouchesReceived() => totalTouchesReceived;
    public bool IsListening() => isListening;
}
