import numpy as np

import pickle

import pandas as pd

import os

from flask import Flask, request, render_template



app = Flask(__name__)



model_path = r'C:\Users\knith\OneDrive\Desktop\python crt\model.pkl'

scaler_path = r'C:\Users\knith\OneDrive\Desktop\python crt\scaler.pkl'



try:

    model = pickle.load(open(model_path, 'rb'))

    scale = pickle.load(open(scaler_path, 'rb'))

except FileNotFoundError as e:

    print(f"File not found: {e}")

    model = None

    scale = None



@app.route('/')

def home():

    return render_template('index.html')



@app.route('/index.html')

def index():

    return render_template('index.html')



@app.route('/predict', methods=["POST", "GET"])

def predict():

    if model is None or scale is None:

        return render_template('index.html', prediction_text="Error loading model or scaler.")

    

    try:

        input_feature = [float(x) for x in request.form.values()]

        names = ['holiday', 'temp', 'rain', 'snow', 'weather',

                 'year', 'month', 'day', 'hours', 'minutes', 'seconds']

        df = pd.DataFrame([input_feature], columns=names)

        df_scaled = scale.transform(df)  # Only transform, not fit_transform

        prediction = model.predict(df_scaled)

        return render_template('index.html', prediction_text=f"Estimated Traffic Volume is: {prediction[0]}")

    except Exception as e:

        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(port=port, debug=True, use_reloader=False)