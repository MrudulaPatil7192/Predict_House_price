import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'linear_model.pkl')

# Exact feature names extracted from your pickled model
FEATURE_COLUMNS = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'number of floors',
    'waterfront present',
    'number of views',
    'condition of the house',
    'grade of the house',
    'Area of the house(excluding basement)',
    'Area of the basement',
    'Built Year',
    'Renovation Year',
    'lot_area_renov',
    'Number of schools nearby',
    'Distance from the airport'
]

model = None
model_status = "Offline"
status_message = ""

# Load the linear regression model using pickle
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        model_status = "Online"
        status_message = "linear_model.pkl loaded successfully."
    except Exception as e:
        model_status = "Error"
        status_message = f"Failed to load linear_model.pkl: {str(e)}"
else:
    model_status = "Missing"
    status_message = "linear_model.pkl not found in root directory."

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #030712;
            --card-bg: rgba(17, 24, 39, 0.85);
            --border: rgba(255, 255, 255, 0.12);
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text-main: #f9fafb;
            --text-sub: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            min-height: 100vh;
            background-color: var(--bg);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Background Animated Glows */
        .glow-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(110px);
            z-index: 0;
            animation: pulseGlow 8s ease-in-out infinite alternate;
        }
        .orb-1 {
            width: 450px;
            height: 450px;
            background: rgba(99, 102, 241, 0.22);
            top: -100px;
            left: -100px;
        }
        .orb-2 {
            width: 400px;
            height: 400px;
            background: rgba(168, 85, 247, 0.18);
            bottom: -100px;
            right: -100px;
            animation-delay: -4s;
        }

        @keyframes pulseGlow {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.15) translate(30px, 20px); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .status-online {
            background: rgba(34, 197, 94, 0.12);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-offline {
            background: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        header h1 {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header p {
            color: var(--text-sub);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        .info-banner {
            margin-bottom: 1.5rem;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            font-size: 0.85rem;
            text-align: center;
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: #c7d2fe;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #fff;
            border: none;
            padding: 1.1rem;
            border-radius: 14px;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px var(--accent-glow);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--accent-glow);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.8rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
            border: 1px solid rgba(129, 140, 248, 0.4);
            border-radius: 20px;
            text-align: center;
            animation: popIn 0.4s ease-out;
        }

        .error-card {
            margin-top: 1.5rem;
            padding: 1rem 1.2rem;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 14px;
            color: #f87171;
            font-size: 0.88rem;
            text-align: center;
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-card h3 {
            font-size: 0.85rem;
            color: #c7d2fe;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-value {
            font-size: 3.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 0.3rem;
            text-shadow: 0 0 30px var(--accent-glow);
        }
    </style>
</head>
<body>
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>

    <div class="container">
        <header>
            <div class="status-badge {% if model_status == 'Online' %}status-online{% else %}status-offline{% endif %}">
                ● Model {{ model_status }}
            </div>
            <h1>House Price Prediction</h1>
            <p>Linear Regression valuation engine based on property metrics</p>
        </header>

        <div class="info-banner">
            {{ status_message }}
        </div>

        <form method="POST" action="/" class="form-grid">
            <div class="input-group">
                <label>Bedrooms</label>
                <input type="number" name="number of bedrooms" value="{{ inputs['number of bedrooms'] if inputs else '3' }}" required>
            </div>

            <div class="input-group">
                <label>Bathrooms</label>
                <input type="number" step="0.25" name="number of bathrooms" value="{{ inputs['number of bathrooms'] if inputs else '2.25' }}" required>
            </div>

            <div class="input-group">
                <label>Living Area (sqft)</label>
                <input type="number" step="any" name="living area" value="{{ inputs['living area'] if inputs else '2570' }}" required>
            </div>

            <div class="input-group">
                <label>Lot Area (sqft)</label>
                <input type="number" step="any" name="lot area" value="{{ inputs['lot area'] if inputs else '7242' }}" required>
            </div>

            <div class="input-group">
                <label>Floors</label>
                <input type="number" step="0.5" name="number of floors" value="{{ inputs['number of floors'] if inputs else '2' }}" required>
            </div>

            <div class="input-group">
                <label>Waterfront Present</label>
                <select name="waterfront present">
                    <option value="0" {% if inputs and inputs['waterfront present'] == '0' %}selected{% endif %}>No (0)</option>
                    <option value="1" {% if inputs and inputs['waterfront present'] == '1' %}selected{% endif %}>Yes (1)</option>
                </select>
            </div>

            <div class="input-group">
                <label>Number of Views</label>
                <input type="number" min="0" max="4" name="number of views" value="{{ inputs['number of views'] if inputs else '0' }}" required>
            </div>

            <div class="input-group">
                <label>House Condition (1-5)</label>
                <input type="number" min="1" max="5" name="condition of the house" value="{{ inputs['condition of the house'] if inputs else '3' }}" required>
            </div>

            <div class="input-group">
                <label>House Grade (1-13)</label>
                <input type="number" min="1" max="13" name="grade of the house" value="{{ inputs['grade of the house'] if inputs else '7' }}" required>
            </div>

            <div class="input-group">
                <label>Area (Excl. Basement)</label>
                <input type="number" step="any" name="Area of the house(excluding basement)" value="{{ inputs['Area of the house(excluding basement)'] if inputs else '2170' }}" required>
            </div>

            <div class="input-group">
                <label>Area of Basement</label>
                <input type="number" step="any" name="Area of the basement" value="{{ inputs['Area of the basement'] if inputs else '400' }}" required>
            </div>

            <div class="input-group">
                <label>Built Year</label>
                <input type="number" name="Built Year" value="{{ inputs['Built Year'] if inputs else '1951' }}" required>
            </div>

            <div class="input-group">
                <label>Renovation Year</label>
                <input type="number" name="Renovation Year" value="{{ inputs['Renovation Year'] if inputs else '1991' }}" required>
            </div>

            <div class="input-group">
                <label>Renovated Lot Area</label>
                <input type="number" step="any" name="lot_area_renov" value="{{ inputs['lot_area_renov'] if inputs else '7639' }}" required>
            </div>

            <div class="input-group">
                <label>Nearby Schools</label>
                <input type="number" name="Number of schools nearby" value="{{ inputs['Number of schools nearby'] if inputs else '3' }}" required>
            </div>

            <div class="input-group">
                <label>Airport Distance (km)</label>
                <input type="number" step="any" name="Distance from the airport" value="{{ inputs['Distance from the airport'] if inputs else '50' }}" required>
            </div>

            <button type="submit" class="btn-submit">
                Calculate Predicted Price
            </button>
        </form>

        {% if error_msg %}
            <div class="error-card">
                {{ error_msg }}
            </div>
        {% endif %}

        {% if prediction_result is not none %}
            <div class="result-card">
                <h3>Estimated House Valuation</h3>
                <div class="prediction-value">${{ "{:,.2f}".format(prediction_result) }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def execute_prediction(data_source):
    """Formats inputs into a DataFrame matching feature names and runs linear regression prediction."""
    if model is None:
        raise ValueError("Linear regression model is not loaded. Ensure linear_model.pkl exists.")
    
    input_dict = {}
    for col in FEATURE_COLUMNS:
        if col not in data_source:
            raise KeyError(f"Missing required feature input: '{col}'")
        input_dict[col] = [float(data_source[col])]

    df_input = pd.DataFrame(input_dict)
    prediction = model.predict(df_input)
    return float(prediction[0])

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    error_msg = None
    inputs = None

    if request.method == 'POST':
        inputs = request.form
        try:
            prediction_result = execute_prediction(inputs)
        except Exception as e:
            error_msg = f"Prediction Error: {str(e)}"

    return render_template_string(
        HTML_LAYOUT, 
        model_status=model_status, 
        status_message=status_message,
        prediction_result=prediction_result,
        error_msg=error_msg,
        inputs=inputs
    )

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        val = execute_prediction(data)
        return jsonify({'prediction': val})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
