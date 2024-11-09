from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Ini akan mengizinkan semua asal untuk mengakses endpoint Anda

# Load model yang telah disimpan
rf_model = joblib.load('random_forest_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Ambil data JSON dari request

        # Memastikan bahwa data memiliki fitur yang dibutuhkan
        required_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400

        # Mengambil nilai fitur
        features = np.array([
            data['age'], data['hypertension'], data['heart_disease'], data['avg_glucose_level']
        ]).reshape(1, -1)

        # Prediksi dari kedua model
        rf_prediction = rf_model.predict(features)
        dt_prediction = dt_model.predict(features)

        # Kembalikan hasil prediksi sebagai JSON
        return jsonify({
            'RandomForest_Prediction': int(rf_prediction[0]),
            'DecisionTree_Prediction': int(dt_prediction[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
