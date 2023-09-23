from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
from flask_cors import CORS  # Import Flask-CORS


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:5000"}})

# Load the model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = joblib.load('models/scaler.pkl')

label_encoders = joblib.load('models/label_encoders.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Define the columns
        numerical_columns = ['Frequency', 'Signal Strength', 'Bandwidth', 'Battery Level']
        categorical_columns = ['Antenna Type', 'Interference Type', 'Device Type', 'Power Source']
        
        # Extract and scale the numerical features
        numerical_data = np.array([data[col] for col in numerical_columns])
        numerical_data_scaled = scaler.transform([numerical_data])[0]

        # Extract and encode the categorical features
        categorical_data_encoded = [label_encoders[col].transform([data[col]])[0] for col in categorical_columns]
        
        # Combine the scaled numerical data and encoded categorical data
        input_data_combined = np.concatenate([numerical_data_scaled, categorical_data_encoded])
        
        # Make a prediction using the trained model
        prediction = model.predict([input_data_combined])
        print("Numerical data shape:", numerical_data.shape)
        print("Scaled numerical data shape:", numerical_data_scaled.shape)
        print("Combined input data shape:", input_data_combined.shape)

        # Return the prediction
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})
