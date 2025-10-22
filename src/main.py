from realsense_camera import RealSenseCamera
import cv2
from image_cropper import ImageCropper
from surface_calibrator import SurfaceCalibrator
from typing import List, Tuple


class MainApplication:
    def __init__(self, width=640, height=480, fps=30):

        # Классы
        self.camera = RealSenseCamera(width, height, fps)
        self.cropper: ImageCropper = None
        self.surface_calibrator: SurfaceCalibrator = None
        # Флаги
        self.running = False
        self.in_crop_mode = False

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
                depth_colormap = self.camera.apply_colormap_to_depth(cropped_depth)
                self._show_frames(cropped_color, depth_colormap)
            else:
                # На случай ошибки — показываем исходное
                depth_colormap = self.camera.apply_colormap_to_depth(self.depth_image)
                self._show_frames(self.color_image, depth_colormap)
        else:
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
            # Теперь можно фильтровать облако точек по высоте над полом
        else:
            print("Не удалось оценить плоскость")

        # calibrator.set_2d_points(points_2d)
        # self.points_3d = calibrator.compute_3d_points()

        # print("3D-координаты (в метрах):")
        # for i, (x, y, z) in enumerate(self.points_3d):
        #     print(f"  Точка {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")

    def _cleanup(self):
        """Очистка ресурсов: остановка камеры и закрытие окон."""
        try:
            self.camera.__exit__(None, None, None)
        except Exception as e:
            print(f"Ошибка при остановке камеры: {e}")
        cv2.destroyAllWindows()
        self.running = False


def main():
    app = MainApplication()
    app.run()


if __name__ == "__main__":
    main()
