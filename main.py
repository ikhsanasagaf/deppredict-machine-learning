from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os

# Inisialisasi aplikasi web Flask
app = Flask(__name__)

# --- Memuat Model dan Scaler saat Aplikasi Dimulai ---
# Ini memastikan model hanya dimuat sekali untuk efisiensi.
try:
    # Menggunakan path relatif agar bekerja di lingkungan deployment
    script_dir = os.path.dirname(__file__)
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    model_path = os.path.join(script_dir, 'model.keras')
    
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = tf.keras.models.load_model(model_path)
    print("Model and scaler loaded successfully!")
except Exception as e:
    # Menangani error jika file tidak ditemukan saat startup
    scaler = None
    model = None
    print(f"CRITICAL: Could not load model or scaler. Error: {e}")

# --- Membuat Endpoint untuk Prediksi ---
# Aplikasi akan "mendengarkan" permintaan POST di alamat /predict
@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah model sudah berhasil dimuat
    if not model or not scaler:
        return jsonify({'error': 'Model atau scaler tidak berhasil dimuat saat startup.'}), 500

    try:
        # Mengambil data JSON dari permintaan yang masuk
        data = request.get_json(force=True)
        # Mengambil array fitur dari data JSON
        features = data['features']

        # Konversi data input menjadi numpy array dan lakukan scaling
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        
        # Lakukan prediksi
        prediction_prob = model.predict(scaled_data)
        # Ubah hasil probabilitas (misal: 0.8) menjadi 0 atau 1
        result = (prediction_prob > 0.5).astype(int)[0][0]
        
        # Mengirim hasil kembali sebagai JSON
        return jsonify({'prediction': int(result)})
    except Exception as e:
        # Mengirim pesan error jika ada masalah dengan data input
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 400

# Rute dasar untuk mengecek apakah server berjalan
@app.route('/', methods=['GET'])
def index():
    return "DepPredict ML API is running!"

# Bagian ini TIDAK akan digunakan oleh Hugging Face,
# tetapi berguna untuk pengujian di komputer lokal Anda.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)