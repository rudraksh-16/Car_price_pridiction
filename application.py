from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Determine base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and data
model_path = os.path.join(BASE_DIR, 'LinearRegressionModel.pkl')
csv_path = os.path.join(BASE_DIR, 'Cleaned car.csv')
model = pickle.load(open(model_path, 'rb'))
car = pd.read_csv(csv_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))

        prediction = model.predict(data)
        output = float(np.round(prediction[0], 2))
        return jsonify({'prediction': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
