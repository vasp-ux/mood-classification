import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# ================= CONFIG ================= #
NUM_CLASSES = 8
# angry, contempt, disgust, fear, happy, neutral, sad, surprise
# ========================================== #

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

# Sanity check (VERY IMPORTANT)
print("Unique labels in y:", np.unique(y))

# One-hot encode labels
y = to_categorical(y, num_classes=NUM_CLASSES)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save model
model.save("emotion_model.keras")
print("✅ Model trained and saved successfully")
