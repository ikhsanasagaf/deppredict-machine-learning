from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

CORS(app, origins=["https://deppredict.netlify.app"])

try:
    script_dir = os.path.dirname(__file__)
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    model_path = os.path.join(script_dir, 'best_model.keras')
    
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = tf.keras.models.load_model(model_path)
    print("Model and scaler loaded successfully!")
except Exception as e:
    scaler = None
    model = None
    print(f"CRITICAL: Could not load model or scaler. Error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model atau scaler tidak berhasil dimuat saat startup.'}), 500

    try:
        data = request.get_json(force=True)
        features = data['features']

        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        
        prediction_prob = model.predict(scaled_data)
        result = (prediction_prob > 0.5).astype(int)[0][0]
        
        return jsonify({'prediction': int(result)})
    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 400

@app.route('/', methods=['GET'])
def index():
    return "DepPredict ML API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)