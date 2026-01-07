import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ================= CONFIG ================= #
NUM_CLASSES = 8   # FER-2013 emotions
IMG_SIZE = 48
# ========================================== #

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

print("Unique labels:", np.unique(y))

# One-hot encode labels
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# Train-test split (IMPORTANT: stratify using original y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= CLASS WEIGHTS ================= #
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)
# ================================================= #

# ================= CNN MODEL ================= #
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
# =============================================== #

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= DATA AUGMENTATION ================= #
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)
# ===================================================== #

# ================= CALLBACKS ================= #
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3
    )
]
# ===================================================== #

# ================= TRAIN MODEL ================= #
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=35,
    callbacks=callbacks,
    class_weight=class_weights
)
# ===================================================== #

# Save model
model.save("emotion_model.keras")
print("✅ Improved model trained and saved successfully")
