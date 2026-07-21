# YOLO Object Detector

Proyek ini merupakan aplikasi pendeteksi objek menggunakan model **YOLOv8 Nano (YOLOv8n)** dari Ultralytics. Program dapat mendeteksi berbagai objek pada gambar dan menampilkan hasil deteksi berupa bounding box, label objek, serta nilai confidence.

---

## 📌 Fitur

- Deteksi objek menggunakan model YOLOv8n
- Mendukung input berupa gambar
- Menampilkan bounding box pada objek yang terdeteksi
- Menampilkan nama kelas (class label)
- Menampilkan nilai confidence setiap objek
- Menyimpan hasil deteksi ke folder output

---

## 🛠️ Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/DerryMarvin14/object_detector.git
cd object_detector
```

### 2. Buat Virtual Environment

Windows

```bash
python -m venv .venv
```

Aktifkan Virtual Environment

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python
```

atau jika menggunakan requirements.txt

```bash
pip install -r requirements.txt
```

---

## ▶️ Cara Menjalankan

Jalankan program dengan:

```bash
python main.py
```

Program akan membaca gambar dari folder **input_images**, melakukan proses deteksi objek menggunakan model YOLOv8n, kemudian menyimpan hasilnya ke folder **output_images**.

---

## ⚙️ Cara Kerja

1. Program memuat model **YOLOv8n** (`yolov8n.pt`).
2. Gambar dibaca menggunakan OpenCV.
3. Model YOLO melakukan proses inferensi untuk mendeteksi objek.
4. Setiap objek yang terdeteksi diberi:
   - Bounding Box
   - Nama objek (Class)
   - Confidence Score
5. Hasil deteksi ditampilkan dan disimpan ke folder **output_images**.

---

## 📂 Struktur Folder

```
object_detector/
│── input_images/
│── output_images/
│── main.py
│── yolov8n.pt
│── README.md
│── .gitignore
```

---

## 🖼️ Tampilan Hasil

Hasil deteksi akan berupa gambar dengan:

- Bounding box pada objek
- Label nama objek
- Nilai confidence

Contoh:

| Sebelum | Sesudah |
|---------|----------|
| Gambar asli | Gambar dengan bounding box, label, dan confidence |

> Tambahkan screenshot hasil deteksi ke folder `images/` atau langsung tampilkan di README.

Contoh:

```markdown
## Contoh Hasil

![Hasil Deteksi](images/hasil_deteksi.png)
```

---

## 📚 Teknologi yang Digunakan

- Python 3
- Ultralytics YOLOv8
- OpenCV
- NumPy

---

## 👤 Author

**Derry Marvin**
