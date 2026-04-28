import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# LSTM TRAINING FILE
# This file:
# 1. loads all saved .npz action files
# 2. combines them into one dataset
# 3. trains an LSTM model for action classification
# 4. saves the model and label mapping

# Dataset folder
dataset_dir = r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\lstm_dataset"

# Action files
dataset_files = [
    "squat_sequences.npz",
    "curl_sequences.npz",
    "handsup_sequences.npz",
    "idle_sequences.npz"
]

# Model save paths
model_save_path = os.path.join(dataset_dir, "lstm_action_model.h5")
label_map_path = os.path.join(dataset_dir, "label_map.json")

# Load all datasets
X_all = []
y_all = []


print("LOADING DATASETS")

for file_name in dataset_files:
    file_path = os.path.join(dataset_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: File not found -> {file_path}")
        continue

    data = np.load(file_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    label_name = str(data["label_name"])
    label_id = int(data["label_id"])

    print(f"Loaded: {file_name}")
    print(f"  Label Name : {label_name}")
    print(f"  Label ID   : {label_id}")
    print(f"  X shape    : {X.shape}")
    print(f"  y shape    : {y.shape}")
    print("--------------------------------------")

    X_all.append(X)
    y_all.append(y)

# Check if data exists
if len(X_all) == 0:
    raise ValueError("No dataset files loaded. Please check dataset paths.")

# Combine all data
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("COMBINED DATASET")
print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)

# Shuffle dataset
X_all, y_all = shuffle(X_all, y_all, random_state=42)


# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)


print("TRAIN / TEST SPLIT")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# LSTM Model
sequence_length = X_train.shape[1]
feature_count = X_train.shape[2]
num_classes = len(np.unique(y_all))


print("MODEL INFO")
print(".....................................")
print("Sequence Length:", sequence_length)
print("Feature Count  :", feature_count)
print("Num Classes    :", num_classes)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, feature_count)),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    model_save_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Train model
print("TRAINING STARTED")
print("...............................")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate model

print("EVALUATION")
print(".......................")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

# Save label map
label_map = {
    "0": "squat",
    "1": "curl",
    "2": "handsup",
    "3": "idle"
}

with open(label_map_path, "w") as f:
    json.dump(label_map, f, indent=4)

print("MODEL SAVED")
print("Best Model Path :", model_save_path)
print("Label Map Path  :", label_map_path)
