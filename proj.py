import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras import backend as K

# -----------------------------
# CHARACTER SET
# -----------------------------
char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
int_to_char = {i + 1: c for i, c in enumerate(char_set)}
num_classes = len(char_set) + 1  # +1 for CTC blank

# -----------------------------
# BUILD CRNN MODEL FOR INFERENCE
# -----------------------------
def build_prediction_model():
    input_img = Input(shape=(32, 128, 1), name="image_input")

    x = Conv2D(64, (3,3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)

    x = Reshape(target_shape=(-1, 128))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)

    y_pred = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_img, outputs=y_pred)
    return model

# -----------------------------
# LOAD MODEL WEIGHTS
# -----------------------------
model_path = "D:/project/handwriting_best.keras"  # your trained model weights
prediction_model = build_prediction_model()
prediction_model.load_weights(model_path)
print(f"✅ Loaded model weights from {model_path}")

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # channel
    img = np.expand_dims(img, axis=0)   # batch
    return img

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_text(image_path):
    img = preprocess_image(image_path)
    preds = prediction_model.predict(img)
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = K.ctc_decode(preds, input_length=input_len)
    decoded_text = ''.join([int_to_char[c] for c in decoded[0][0].numpy() if c > 0])
    return decoded_text

# -----------------------------
# TKINTER GUI
# -----------------------------
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            text = predict_text(file_path)
            messagebox.showinfo("Prediction", f"Predicted text: {text}")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

root = tk.Tk()
root.title("Handwritten Text Recognition")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20, height=2)
upload_btn.pack(pady=20)

exit_btn = tk.Button(root, text="Exit", command=root.quit, width=20, height=2)
exit_btn.pack(pady=20)

root.mainloop()
