from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

try:
    script_dir = os.path.dirname(__file__)
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    model_path = os.path.join(script_dir, 'model.keras')
    
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    scaler = None
    model = None
    print(f"Error loading model/scaler: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model atau scaler tidak berhasil dimuat'}), 500

    try:
        data = request.get_json(force=True)
        features = data['features']

        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        
        prediction_prob = model.predict(scaled_data)
        result = (prediction_prob > 0.5).astype(int)[0][0]
        
        return jsonify({'prediction': int(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return "DepPredict ML API is running!"

if __name__ == '__main__':
    app.run(debug=True, port=8080)