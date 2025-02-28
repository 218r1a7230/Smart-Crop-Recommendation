import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode

# Function to preprocess dataset
def preprocessDataset(dataset):
    le = LabelEncoder()
    dataset.fillna(0, inplace=True)

    X = dataset.drop(['label'], axis=1).values
    Y = dataset['label'].values
    Y = le.fit_transform(Y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    return x_train, x_test, y_train, y_test, le, scaler

# Function to train models
def train_models(x_train, y_train, x_test, y_test):
    models = []

    # Train RandomForest model
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    models.append(rf)

    # Train GradientBoosting model
    gb = GradientBoostingClassifier()
    gb.fit(x_train, y_train)
    models.append(gb)

    # Train SVM model
    svm = SVC()
    svm.fit(x_train, y_train)
    models.append(svm)

    # Train NaiveBayes model
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    models.append(nb)

    # Train MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=0)
    mlp.fit(x_train, y_train)
    models.append(mlp)

    return models

# Function to predict new data
def predict(models, le, scaler, input_values):
    input_values = np.array(input_values).astype(float).reshape(1, -1)
    input_values = scaler.transform(input_values)

    preds = [model.predict(input_values)[0] for model in models]
    vote_result = mode(preds, keepdims=True)[0][0]
    return le.inverse_transform([vote_result])[0]