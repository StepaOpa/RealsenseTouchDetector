from realsense_camera import RealSenseCamera
import cv2
from image_cropper import ImageCropper
from surface_calibrator import SurfaceCalibrator
from touch_processor import (
    TouchProcessor,
)  # Предполагаем, что класс в файле touch_processor.py
from typing import List, Tuple


class MainApplication:
    def __init__(self, width=640, height=480, fps=30):

        # Классы
        self.camera = RealSenseCamera(width, height, fps)
        self.cropper: ImageCropper = None
        self.surface_calibrator: SurfaceCalibrator = None
        self.touch_processor: TouchProcessor = None  # <-- НОВОЕ

        # Флаги
        self.running = False
        self.in_crop_mode = False
        self.surface_calibrated = False  # <-- НОВОЕ: флаг успешной калибровки

        # Сырой поток
        self.color_image = None
        self.depth_image = None

        # Обрезанный поток
        self.cropped_color_image = None
        self.cropped_depth_image = None

        # Списки точек
        self.points_2d = []
        self.points_3d = []

    def run(self):
        """Основной цикл приложения."""
        if not self._start_camera():
            return

        self.running = True
        try:
            while self.running:
                if not self._process_frames():
                    continue
                self._handle_user_input()
        finally:
            self._cleanup()

    def _start_camera(self) -> bool:
        """Попытка запуска камеры через контекстный менеджер."""
        try:
            self.camera.__enter__()
            return True
        except RuntimeError as e:
            print(f"Ошибка запуска камеры: {e}")
            return False

    def _process_frames(self) -> bool:
        self.color_image, self.depth_image = self.camera.get_frames()
        if self.color_image is None or self.depth_image is None:
            return False

        if self.in_crop_mode and self.cropper is not None:
            # Обрезаем КАЖДЫЙ новый кадр в реальном времени!
            cropped_color = self.cropper.crop_image(self.color_image)
            cropped_depth = self.cropper.crop_image(self.depth_image)

            if cropped_color is not None and cropped_depth is not None:
                # === НОВОЕ: Обработка касаний на кропнутом кадре ===
                if self.surface_calibrated and self.touch_processor:
                    # Обновляем интринсики, если они изменились (опционально)
                    intrinsics, _ = self.camera.get_intrinsics()
                    if intrinsics:
                        self.touch_processor.set_intrinsics(intrinsics)

                    # Обработка кропнутого кадра
                    touches = self.touch_processor.process_frame(cropped_depth)
                    # touches = [(x, y), ...] в пикселях кропнутого изображения
                    print(f"Касания в кропнутом кадре: {touches}")

                depth_colormap = self.camera.apply_colormap_to_depth(cropped_depth)
                self._show_frames(cropped_color, depth_colormap)
            else:
                # На случай ошибки — показываем исходное
                depth_colormap = self.camera.apply_colormap_to_depth(self.depth_image)
                self._show_frames(self.color_image, depth_colormap)
        else:
            # === Обработка на полном кадре, если не в режиме кропа ===
            if self.surface_calibrated and self.touch_processor:
                intrinsics, _ = self.camera.get_intrinsics()
                if intrinsics:
                    self.touch_processor.set_intrinsics(intrinsics)
                touches = self.touch_processor.process_frame(self.depth_image)
                # print(f"Касания в полном кадре: {touches}")

            depth_colormap = self.camera.apply_colormap_to_depth(self.depth_image)
            self._show_frames(self.color_image, depth_colormap)

        return True

    def _show_frames(self, color_image, depth_image):
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image)

    def _handle_user_input(self):
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc — выход
            self.running = False

        elif key == ord("c") or key == ord("C"):
            self._crop_images()

        elif key == ord("v") or key == ord("V"):
            self.in_crop_mode = not self.in_crop_mode

        elif key == ord("s") or key == ord("S"):
            self.calibrate_surface(self.points_2d)

    def _crop_images(self):
        if self.color_image is None:
            print("Нет кадра для выбора области.")
            return

        # Создаём временный кроппер для выбора точек
        temp_cropper = ImageCropper(self.color_image.copy(), output_size=(640, 480))
        temp_cropper.select_crop_region()

        if len(temp_cropper.points) == 4:
            # Сохраняем кроппер — он содержит точки и знает, как кропать
            self.cropper = temp_cropper
            self.points_2d = self.cropper.points
            self.in_crop_mode = True
            print("Режим кроппинга потока ВКЛЮЧЁН.")
        else:
            print("Кроппинг отменён.")

    def calibrate_surface(self, points_2d: List[Tuple[int, int]] = []):
        calibrator = SurfaceCalibrator(self.camera)

        calibrator.set_2d_points(points_2d)

        # Оцениваем плоскость с RANSAC
        plane = calibrator.estimate_floor_plane_with_ransac(
            roi_radius=60, distance_threshold=0.03, num_iterations=1500  # 3 см
        )

        if plane:
            print("Плоскость пола:", plane)
            # === ИНИЦИАЛИЗАЦИЯ TouchProcessor ===
            self.touch_processor = TouchProcessor()
            self.touch_processor.set_floor_plane(plane)

            # Устанавливаем интринсики
            intrinsics, _ = self.camera.get_intrinsics()
            if intrinsics:
                self.touch_processor.set_intrinsics(intrinsics)

            # Настройки фильтрации (пример)
            self.touch_processor.set_height_range(0.0, 0.15)  # 0-15 см над плоскостью
            self.touch_processor.set_area_range(200, 2000)  # фильтр по площади

            self.surface_calibrated = True
            print("TouchProcessor инициализирован и готов к работе.")
            print("Бинарный поток отображается в окне 'Binary Touch Detection'.")
        else:
            print("Не удалось оценить плоскость")
            self.surface_calibrated = False

    def _cleanup(self):
        """Очистка ресурсов: остановка камеры и закрытие окон."""
        try:
            self.camera.__exit__(None, None, None)
        except Exception as e:
            print(f"Ошибка при остановке камеры: {e}")

        # Закрытие окон TouchProcessor (если был инициализирован)
        if self.touch_processor:
            self.touch_processor.close()  # Вызываем метод close, если он есть в вашем TouchProcessor

        cv2.destroyAllWindows()
        self.running = False


def main():
    app = MainApplication()
    app.run()


if __name__ == "__main__":
    main()
