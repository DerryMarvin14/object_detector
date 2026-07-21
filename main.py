import argparse
import os
import shutil
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
DEFAULT_OUTPUT = "output_images"
DEFAULT_INPUT_FOLDER = "input_images"


def parse_args():
    parser = argparse.ArgumentParser(description="Deteksi objek dengan YOLOv8")
    parser.add_argument(
        "--source",
        default=DEFAULT_INPUT_FOLDER,
        help="Sumber input: folder, file gambar, atau webcam ('webcam' atau '0').",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="File model YOLO (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Folder output untuk menyimpan hasil deteksi.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold untuk deteksi.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tampilkan hasil deteksi di layar (berguna untuk webcam).",
    )
    return parser.parse_args()


def ensure_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. Pastikan file yolov8n.pt ada di folder proyek."
        )


def prepare_input_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        print(
            f"Folder input_images dibuat: {folder}\nMasukkan gambar (.png, .jpg, .jpeg) ke folder ini, lalu jalankan ulang."
        )
        return False
    return True


def is_image_file(path):
    return str(path).lower().endswith((".png", ".jpg", ".jpeg"))


def show_image_window(window_name, image, delay=1):
    cv2.imshow(window_name, image)
    return cv2.waitKey(delay) & 0xFF


def detect_webcam(model, conf):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka webcam. Pastikan webcam terpasang dan tidak digunakan aplikasi lain.")

    cv2.namedWindow("Deteksi Webcam", cv2.WINDOW_NORMAL)
    print("Tekan 'q' untuk keluar dari tampilan webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)
        annotated = frame
        if results:
            annotated = results[0].plot()

        key = show_image_window("Deteksi Webcam", annotated, delay=1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_image_or_folder(model, source_arg, output_folder, conf, show_display):
    if os.path.isdir(source_arg):
        image_files = [
            os.path.join(source_arg, f)
            for f in os.listdir(source_arg)
            if is_image_file(f)
        ]
        if not image_files:
            print(f"Tidak ada file gambar di folder {source_arg}.")
            return
        print(f"Mendeteksi semua gambar di folder: {source_arg}")
        for image_path in image_files:
            results = model(image_path, conf=conf, save=True, save_dir=output_folder)
            if show_display and results:
                annotated = results[0].plot()
                print(f"Menampilkan hasil untuk {image_path}")
                if show_image_window("Deteksi Gambar", annotated, delay=0) == ord("q"):
                    break
    else:
        print(f"Mendeteksi gambar: {source_arg}")
        results = model(source_arg, conf=conf, save=True, save_dir=output_folder)
        if results and show_display:
            annotated = results[0].plot()
            print(f"Menampilkan hasil untuk {source_arg}")
            show_image_window("Deteksi Gambar", annotated, delay=0)
            cv2.destroyAllWindows()


def object_detector(source, model_path, output_folder, conf, show_display=False):
    model_path = os.path.abspath(model_path)
    ensure_model(model_path)

    if isinstance(source, str) and source.lower() in ("webcam", "0"):
        is_webcam = True
        source_arg = 0
        print("Mendeteksi dari webcam...")
    else:
        is_webcam = False
        source_arg = source

    if isinstance(source_arg, str) and os.path.isdir(source_arg):
        if not prepare_input_folder(source_arg):
            return
    elif isinstance(source_arg, str) and not os.path.exists(source_arg):
        raise FileNotFoundError(
            f"Sumber tidak ditemukan: {source_arg}. Gunakan folder, file gambar, atau 'webcam'."
        )
    elif isinstance(source_arg, str) and os.path.isfile(source_arg):
        if not is_image_file(source_arg):
            raise ValueError(
                f"File tidak valid: {source_arg}. Gunakan file .png, .jpg, atau .jpeg."
            )
    
    try:
        model = YOLO(model_path)
    except PermissionError as exc:
        raise PermissionError(
            "File model yolov8n.pt sedang digunakan oleh proses lain. Tutup program lain yang membuka file tersebut, lalu jalankan ulang."
        ) from exc

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    show = show_display or is_webcam
    print(f"Menjalankan deteksi. Hasil disimpan di folder: {output_folder}")
    print(f"Tampilan langsung: {'aktif' if show else 'mati'}")

    if is_webcam:
        detect_webcam(model, conf)
    else:
        detect_image_or_folder(model, source_arg, output_folder, conf, show)

    if os.path.isdir(output_folder):
        saved = [f for f in os.listdir(output_folder) if is_image_file(f)]
        if saved:
            print(f"Selesai. File hasil disimpan: {len(saved)} file di {output_folder}")
        else:
            print(f"Selesai. Tidak ada file gambar hasil di {output_folder}. Pastikan deteksi berjalan dan file input valid.")


if __name__ == "__main__":
    args = parse_args()
    object_detector(args.source, args.model, args.output, args.conf, args.show)
