import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Dense, Bidirectional, LSTM, Permute
)
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image as keras_image
import cv2
import os

# ===============================
# PATHS
# ===============================
# Replace with your paths
TRAIN_MODEL_PATH = r"D:/project/htr_ctc_words_multilang_best.keras"
MODEL_WEIGHTS_PATH = TRAIN_MODEL_PATH  # weights are saved inside
IMG_HEIGHT = 32
IMG_WIDTH = 256

# ===============================
# LOAD TRAINING MODEL TO EXTRACT CHARSET
# ===============================
print("Loading training model to extract charset...")
# Use compile=False to avoid CTC Lambda issues
training_model = load_model(TRAIN_MODEL_PATH, compile=False)

# Extract number of output chars
y_pred_layer = training_model.get_layer("y_pred")
num_classes = y_pred_layer.output_shape[-1]  # includes blank

print("Number of output classes (with blank):", num_classes)

# ===============================
# BUILD PREDICTION MODEL (without CTC)
# ===============================
image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

x = Conv2D(64, (3, 3), padding="same", activation="relu")(image_input)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2))(x)

x = BatchNormalization()(x)

x = Permute((2, 1, 3))(x)
x = Reshape((64, 8 * 128))(x)

x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)

y_pred = Dense(num_classes, activation="softmax", name="y_pred")(x)

prediction_model = Model(inputs=image_input, outputs=y_pred)
print("Prediction model built.")

# ===============================
# LOAD WEIGHTS
# ===============================
print("Loading weights from training model...")
prediction_model.load_weights(MODEL_WEIGHTS_PATH, skip_mismatch=True)
print("Weights loaded successfully.")

# ===============================
# CHARSET (use the same as training)
# ===============================
# Ideally you saved charset separately; here we extract from training model
# For example purposes, let's assume lowercase + uppercase + digits
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_idx = {c: i for i, c in enumerate(charset)}
idx_to_char = {i: c for i, c in enumerate(charset)}
blank_idx = len(charset)

# ===============================
# PREDICTION FUNCTIONS
# ===============================
def preprocess_image(img_path):
    """Load and resize a word image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

def decode_prediction(pred):
    """CTC greedy decoding"""
    pred_text = ""
    pred_indices = np.argmax(pred, axis=-1)[0]  # (time_steps,)
    prev_idx = -1
    for idx in pred_indices:
        if idx != prev_idx and idx != blank_idx:
            pred_text += idx_to_char.get(idx, "")
        prev_idx = idx
    return pred_text

def predict_word(img_path):
    img = preprocess_image(img_path)
    pred = prediction_model.predict(img)
    text = decode_prediction(pred)
    return text

# ===============================
# EXAMPLE USAGE
# ===============================
if __name__ == "__main__":
    # Select a folder of word images
    FOLDER_PATH = r"D:/project/test_words"
    for fname in os.listdir(FOLDER_PATH):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(FOLDER_PATH, fname)
            text = predict_word(full_path)
            print(f"{fname} -> {text}")