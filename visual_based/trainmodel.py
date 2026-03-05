import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ================= PATH ================= #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= LOAD DATA ================= #

X = np.load(os.path.join(BASE_DIR, "X.npy"))
y = np.load(os.path.join(BASE_DIR, "y.npy"))

NUM_CLASSES = 8
IMG_SIZE = 48

print("✅ Data loaded")
print("X shape:", X.shape)
print("Unique labels:", np.unique(y))

# ================= ONE-HOT ================= #

y = to_categorical(y, NUM_CLASSES)

# ================= SPLIT ================= #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= MODEL ================= #

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

# ================= COMPILE ================= #

model.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN ================= #

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64
)

# ================= SAVE ================= #

MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
model.save(MODEL_PATH)

print("\n✅ FER-only model trained & saved successfully")
print("📁 Saved at:", MODEL_PATH)
