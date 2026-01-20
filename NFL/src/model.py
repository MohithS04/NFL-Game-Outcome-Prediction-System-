import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
import joblib
import os
from .data import load_data
from .features import prepare_features

MODEL_PATH = 'model.pkl'

def train_model():
    """
    Trains a Logistic Regression model on historical NFL data.
    """
    # Load data from 2014 to 2025 for training
    years = list(range(2014, 2026))
    schedule = load_data(years)
    
    X, y, _ = prepare_features(schedule)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    
    print(f"Model trained. Accuracy: {acc:.4f}, Brier Score: {brier:.4f}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load_trained_model():
    """
    Loads the trained model from disk.
    If not found, trains a new one.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        print("Model not found, training new model...")
        return train_model()

def predict_game(model, features):
    """
    Predicts win probability for home team.
    """
    # ensure features are in correct order/shape
    prob = model.predict_proba([features])[0][1]
    return prob

if __name__ == "__main__":
    train_model()
