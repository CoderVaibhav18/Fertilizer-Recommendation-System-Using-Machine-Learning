from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🔄 Load the trained model and encoders
model = joblib.load('fertilizer_model.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')

@app.route('/')
def home():
    return "🚜 Fertilizer Recommendation API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 🧾 Extract input features
        temperature = data['Temparature']
        humidity = data['Humidity']
        moisture = data['Moisture']
        soil_type = soil_encoder.transform([data['Soil Type']])[0]
        crop_type = crop_encoder.transform([data['Crop Type']])[0]
        nitrogen = data['Nitrogen']
        potassium = data['Potassium']
        phosphorous = data['Phosphorous']

        # 🔢 Create input array
        input_data = np.array([[temperature, humidity, moisture, soil_type,
                                crop_type, nitrogen, potassium, phosphorous]])

        # 🎯 Predict
        prediction = model.predict(input_data)
        predicted_fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

        return jsonify({
            'recommended_fertilizer': predicted_fertilizer
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
