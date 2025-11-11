# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Sample dataset: Iris (built-in)
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to models folder
joblib.dump(model, "models/model.pkl")
print("✅ Model trained and saved as models/model.pkl")
