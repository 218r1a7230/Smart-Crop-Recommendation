from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from model import preprocessDataset, train_models, predict

app = Flask(__name__)

# Load dataset and preprocess
filename = "Crop_recommendation.csv"
dataset = pd.read_csv(filename)
x_train, x_test, y_train, y_test, le, scaler = preprocessDataset(dataset)
models = train_models(x_train, y_train, x_test, y_test)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        user_input = [float(x) for x in request.form.values()]
        prediction = predict(models, le, scaler, user_input)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)