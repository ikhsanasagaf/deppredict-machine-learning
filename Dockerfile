# Tahap 1: Menentukan Base Image
# Kita mulai dengan image Python 3.9 'bullseye' yang modern dan stabil
# untuk menghindari masalah OpenSSL.
FROM python:3.9-bullseye

# Tahap 2: Menyiapkan Lingkungan Kerja
# Membuat dan menetapkan direktori /app di dalam container.
WORKDIR /app

# Tahap 3: Mengoptimalkan Cache dengan Menyalin requirements.txt terlebih dahulu
# Ini mempercepat proses build jika Anda hanya mengubah kode tanpa menambah paket baru.
COPY requirements.txt requirements.txt

# Tahap 4: Menginstall Semua Dependency
# Menjalankan pip untuk menginstall semua paket yang dibutuhkan.
RUN pip install --no-cache-dir -r requirements.txt

# Tahap 5: Menyalin Seluruh Kode Aplikasi ke Dalam Container
COPY . .

# Tahap 6: Memberi Tahu Port Mana yang Digunakan
# Port yang akan digunakan oleh Gunicorn untuk menjalankan aplikasi.
EXPOSE 8080

# Tahap 7: Menjalankan Aplikasi
# Perintah final yang akan dieksekusi saat container dimulai.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]