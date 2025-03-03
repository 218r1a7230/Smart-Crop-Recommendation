from flask import Flask, request, render_template, session, redirect, url_for
import numpy as np
import pandas as pd
from model import preprocessDataset, train_models, predict, save_plots
import joblib
import os
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session management

# Crop translation dictionary
crop_translation = {
    "rice": "బియ్యం (Biyyam)", "maize": "మొక్కజొన్న (Mokkajonna)",
    "chickpea": "శనగ (Shanaga)", "kidneybeans": "రాజ్మా (Rajma)",
    "pigeonpeas": "కందులు (Kandulu)", "mothbeans": "బోబ్బర్లు (bobbarlu)",
    "mungbean": "పెసలు (Pesalu)", "blackgram": "మినుములు (Minumulu)",
    "lentil": "మసూర్ పప్పు (Masoor Pappu)", "pomegranate": "దానిమ్మ (Danimma)",
    "banana": "అరటి (Arati)", "mango": "మామిడి (Mamidi)",
    "grapes": "ద్రాక్ష (Draksha)", "watermelon": "పుచ్చకాయ (Puchchakaya)",
    "muskmelon": "ఖర్బుజ్జ (Kharbuja)", "apple": "సెపు (Sepu)",
    "orange": "కమలాపండు (Kamalapandu)", "papaya": "బొప్పాయి (Boppayi)",
    "coconut": "కొబ్బరి (Kobbari)", "cotton": "పత్తి (Pathi)",
    "jute": "జనపనార (Janapanaara)", "coffee": "కాఫీ (Coffee)"
}

# Load dataset and preprocess
filename = "Crop_recommendation.csv"
dataset = pd.read_csv(filename)
x_train, x_test, y_train, y_test, le, scaler, dataset = preprocessDataset(dataset)

# Ensure the saved_models folder exists
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")


# Load or train the hybrid model
if os.path.exists("saved_models/hybrid_model.pkl"):
    try:
        hybrid_model = joblib.load("saved_models/hybrid_model.pkl")
        le = joblib.load("saved_models/label_encoder.pkl")
        scaler = joblib.load("saved_models/scaler.pkl")
        
        # ✅ Load accuracies if available
        if os.path.exists("saved_models/accuracies.pkl"):
            accuracies = joblib.load("saved_models/accuracies.pkl")
        else:
            accuracies = None  # Handle case where accuracies.pkl is missing

    except Exception as e:
        print(f"Model loading failed: {e}, retraining model...")
        hybrid_model, accuracies = train_models(x_train, y_train, x_test, y_test, dataset, le, scaler)
else:
    print("No pre-trained model found, training a new model...")
    hybrid_model, accuracies = train_models(x_train, y_train, x_test, y_test, dataset, le, scaler)

# Ensure plots are saved if not already available
plots_path = "static/plots"
if not os.path.exists(plots_path) or len(os.listdir(plots_path)) == 0:
    save_plots(dataset, y_test, hybrid_model.predict(x_test), hybrid_model, accuracies)

@app.route('/')
def index():
    session.clear()  # Clears all previous session data
    return redirect(url_for('step'))

@app.route('/step', methods=['GET', 'POST'])
def step():
    session.setdefault('step', 1)
    step = session['step']

    if step > 7:
        return redirect(url_for('predict_crop'))

    if request.method == 'POST':
        value = request.form.get(f'step_{step}', "").strip()
        try:
            value = float(value)
            if value < 0:
                raise ValueError("Negative values are not allowed.")
        except ValueError:
            return render_template('index.html', step=step, error="Please enter a valid positive number.")

        session[f'step_{step}'] = value
        session['step'] += 1
        return redirect(url_for('step'))

    return render_template('index.html', step=step)

@app.route('/back')
def go_back():
    if session.get('step', 1) > 1:
        session['step'] -= 1
    return redirect(url_for('step'))

@app.route('/predict')
def predict_crop():
    if not hybrid_model:
        return "Error: Model not loaded. Please restart the server."

    input_values = [float(session.get(f'step_{i}', 0) or 0) for i in range(1, 8)]
    
    try:
        english_prediction = predict(hybrid_model, le, scaler, input_values)
        telugu_prediction = crop_translation.get(english_prediction, "Unknown")
    except Exception as e:
        return f"Prediction failed: {e}"

    session['step'] = 8
    return render_template('index.html', step=8, prediction=f"{english_prediction} ({telugu_prediction})", accuracy=accuracies["Hybrid"] if accuracies else "N/A")

if __name__ == '__main__':
    app.run(debug=True)
