# DepPredict-Capstone-Project

Proyek ini menerapkan model pembelajaran mendalam untuk memprediksi depresi siswa berdasarkan berbagai faktor akademis, sosial, dan pribadi. Model ini dibangun menggunakan TensorFlow/Keras dan diterapkan untuk inferensi web menggunakan TensorFlow.js.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Model Architecture](#model-architecture)
4.  [Prerequisites](#prerequisites)
5.  [Setup and Installation](#setup-and-installation)
6.  [Project Structure](#project-structure)
7.  [How to Run/Use](#how-to-runuse)
    * [Local Development (Python)](#local-development-python)
    * [Web Deployment (TensorFlow.js)](#web-deployment-tensorflowjs)
8.  [Inference Example](#inference-example)
9.  [Contact](#contact)

## 1. Project Overview

Repositori ini berisi kode untuk melatih model jaringan saraf guna memprediksi depresi pada siswa. Fitur utama dari proyek ini meliputi:
* Prapemrosesan data, termasuk penanganan nilai yang hilang, pengodean fitur kategoris, dan deteksi outlier.
* Pelatihan model menggunakan TensorFlow/Keras.
* Konversi model ke format TensorFlow.js untuk kompatibilitas peramban web.
* Mengekspor parameter `StandardScaler` untuk prapemrosesan data yang konsisten selama inferensi web.

## 2. Dataset

Dataset yang digunakan untuk melatih model ini bersumber dari Kaggle:
**Link:** [Kumpulan Data Depresi Mahasiswa](https://www.kaggle.com/datasets/hopesb/student-depression-dataset/data)

Kumpulan data ini mencakup berbagai fitur seperti usia, tekanan akademis, IPK, durasi tidur, kebiasaan makan, tekanan finansial, riwayat keluarga dengan penyakit mental, dan banyak lagi.

## 3. Model Architecture

Model ini adalah model Keras Sequential, kemungkinan terdiri dari lapisan Padat, yang dirancang untuk klasifikasi biner (tertekan/tidak tertekan). Arsitektur spesifik (jumlah lapisan, neuron, fungsi aktivasi) didefinisikan dalam buku catatan `DepPredict_Model_(2).ipynb`.

## 4. Prerequisites

Sebelum menjalankan kode, pastikan Anda telah menginstal yang berikut ini:

* Python 3.8+
* pip (Python package installer)

## 5. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    *(Note: Replace `<repository_url>` and `<repository_name>` with your actual repository details if this is a Git repo. If not, this section is for local setup after you've downloaded the files.)*

2.  **Install Python dependencies:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow tensorflowjs seaborn matplotlib jupyter h5py
    ```

## 6. Project Structure

.
├── DepPredict_Model_(2).ipynb  # Jupyter Notebook containing model training, preprocessing, and conversion steps.
├── best_model.keras            # Trained Keras model in native .keras format.
├── model_predict.h5            # Trained Keras model in legacy .h5 format (alternative).
├── scaler.pkl                  # Python pickle file containing the fitted StandardScaler object.
├── scaler_params.json          # JSON file containing mean and scale parameters from the StandardScaler, for web use.
├── Student Depression Dataset.csv # The input dataset.
├── model_web_from_h5/          # Directory containing the converted TensorFlow.js model files.
│   ├── model.json              # Model architecture (for TensorFlow.js).
│   └── group1-shard1of1.bin    # Model weights (for TensorFlow.js, might be multiple .bin files).
└── README.md                   # This README file.


## 7. How to Run/Use

### Local Development (Python)

To re-run the training, preprocessing, and conversion steps:

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook DepPredict_Model_(2).ipynb
    ```
2.  **Jalankan semua sel:** Jalankan semua sel di buku catatan secara berurutan. Ini akan:
    * Memuat dan memproses data terlebih dahulu.
    * Melatih model pembelajaran mendalam.
    * Menyimpan model yang telah dilatih sebagai `best_model.keras` (dan `model_predict.h5`).
    * Menyimpan objek `StandardScaler` sebagai `scaler.pkl`.
    * Mengekspor parameter `StandardScaler` ke `scaler_params.json`.
    * Mengonversi model ke format TensorFlow.js (file `model.json` dan `.bin`) di direktori `model_web_from_h5/`.

### Web Deployment (TensorFlow.js)

To use the model in a web application:

1.  **Transfer the converted files:** Copy the `model_web_from_h5/` folder and `scaler_params.json` file into your web project's directory.
2.  **Load the model in JavaScript:** Use the TensorFlow.js library to load the `model.json` file.
3.  **Apply scaling:** Use the `scaler_params.json` (mean and scale values) to manually preprocess new input data in JavaScript, mirroring the `StandardScaler` operation in Python.
4.  **Make predictions:** Feed the preprocessed data into the loaded TensorFlow.js model for inference.

Example of loading the model in JavaScript (simplified):

```javascript
// In your web application's JavaScript
import * as tf from '@tensorflow/tfjs';

async function loadAndPredict(inputData) {
    // 1. Load the model
    const model = await tf.loadLayersModel('./model_web_from_h5/model.json');

    // 2. Load scaler parameters (from scaler_params.json)
    const scalerParamsResponse = await fetch('./scaler_params.json');
    const scalerParams = await scalerParamsResponse.json();
    const mean = tf.tensor1d(scalerParams.mean);
    const scale = tf.tensor1d(scalerParams.scale);

    // 3. Preprocess new input data in JavaScript (example - actual preprocessing might be more complex)
    //    Ensure inputData is a TF.Tensor of shape [1, num_features]
    let preprocessedInput = tf.tensor2d([inputData]); // Example: convert array to tensor
    preprocessedInput = preprocessedInput.sub(mean).div(scale);

    // 4. Make a prediction
    const prediction = model.predict(preprocessedInput);
    const probability = prediction.dataSync()[0]; // Get the raw prediction value

    if (probability > 0.5) {
        console.log(`Terindikasi Depresi (Probabilitas: ${(probability * 100).toFixed(2)}%)`);
        return `Terindikasi Depresi (Probabilitas: ${(probability * 100).toFixed(2)}%)`;
    } else {
        console.log(`Tidak Terindikasi Depresi (Probabilitas: ${(probability * 100).toFixed(2)}%)`);
        return `Tidak Terindikasi Depresi (Probabilitas: ${(probability * 100).toFixed(2)}%)`;
    }
}

// Example usage (replace with actual new student data)
// Make sure this matches the feature order and preprocessing steps from your Python notebook
const sampleInputFeatures = [
    0, // id (if used as a feature, otherwise remove)
    22.0, // Age
    3.0,  // Academic Pressure
    0.0,  // Work Pressure
    7.5,  // CGPA (original scale 0-10, will be scaled to 1-4 internally by your JS preprocessor)
    4.0,  // Study Satisfaction
    0.0,  // Job Satisfaction
    3.0,  // Sleep Duration (mapped from '7-8 hours')
    3.0,  // Dietary Habits (mapped from 'Healthy')
    0.0,  // Have you ever had suicidal thoughts ? (mapped from 'No')
    6.0,  // Work/Study Hours
    2.0,  // Financial Stress
    0.0   // Family History of Mental Illness (mapped from 'No')
];

// loadAndPredict(sampleInputFeatures);
