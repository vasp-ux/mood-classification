import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Load data from PROJECT ROOT
PROJECT_ROOT = os.path.dirname(BASE_DIR)

X = np.load(os.path.join(PROJECT_ROOT, "X.npy"))
y = np.load(os.path.join(PROJECT_ROOT, "y.npy"))

# Evaluate
loss, accuracy = model.evaluate(X, y)
print("Final Model Accuracy:", accuracy)
