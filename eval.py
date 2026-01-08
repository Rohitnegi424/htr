import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Dense, Bidirectional, LSTM, Permute
)
from tensorflow.keras.models import Model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# =============================== CHARSET ===============================
data_path = r"D:/project/processed_data/data.npz"
data = np.load(data_path, allow_pickle=True)
texts = data["labels"]

charset = sorted(set("".join(texts)))
char_to_idx = {c: i for i, c in enumerate(charset)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
num_chars = len(charset)
blank_idx = num_chars

print("Charset loaded. Number of chars:", num_chars)

# =============================== INFERENCE MODEL ===============================
def build_inference_model():
    image_input = Input(shape=(32, 256, 1), name="image")
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Permute((2, 1, 3))(x)
    x = Reshape((64, 8 * 128))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    y_pred = Dense(num_chars + 1, activation="softmax", name="y_pred")(x)
    return Model(inputs=image_input, outputs=y_pred)

weights_path = r"D:/project/htr_ctc_words_multilang_best.keras"
model = build_inference_model()
model.load_weights(weights_path)
print("Model and weights loaded successfully!")

# =============================== PREPROCESS IMAGE ===============================
def preprocess_image(img_path):
    """
    Preprocess the input image exactly like training:
    - Grayscale
    - Scale height to 32, scale width proportionally
    - Pad width to 256
    Returns:
    - model_input: ready for prediction
    - display_img: PIL Image of padded input for display
    """
    IMG_HEIGHT = 32
    MAX_WIDTH = 256

    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Resize proportionally to height
    h, w = img_np.shape
    scale = IMG_HEIGHT / h
    new_w = int(w * scale)
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH

    img_np = np.array(Image.fromarray(img_np).resize((new_w, IMG_HEIGHT), Image.BILINEAR))

    # Pad width to MAX_WIDTH
    padded = np.zeros((IMG_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img_np

    # Model input
    model_input = np.expand_dims(padded, axis=-1)  # channel
    model_input = np.expand_dims(model_input, axis=0)   # batch

    # Image for display
    display_img = Image.fromarray((padded*255).astype(np.uint8))

    return model_input, display_img

# =============================== DECODE PREDICTION ===============================
def decode_prediction(pred):
    pred_indices = np.argmax(pred, axis=-1)[0]
    text = ""
    prev = -1
    for i in pred_indices:
        if i != prev and i != blank_idx:
            text += idx_to_char[i]
        prev = i
    return text

# =============================== PREDICT FUNCTION ===============================
def predict_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        return
    try:
        # Preprocess for model & get padded image for display
        img_input, padded_img = preprocess_image(file_path)

        # Display the original selected image
        orig_img = Image.open(file_path)
        orig_img.thumbnail((400, 100))
        orig_img_tk = ImageTk.PhotoImage(orig_img)
        original_label.config(image=orig_img_tk)
        original_label.image = orig_img_tk

        # Display the padded image (model input)
        padded_img = padded_img.resize((400, 50))  # scale for display
        padded_img_tk = ImageTk.PhotoImage(padded_img)
        padded_label.config(image=padded_img_tk)
        padded_label.image = padded_img_tk

        # Prediction
        pred = model.predict(img_input)
        text = decode_prediction(pred)
        result_var.set(f"Predicted Text: {text}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# =============================== GUI ===============================
root = tk.Tk()
root.title("HTR Prediction GUI")

tk.Button(root, text="Select Image & Predict", command=predict_image).pack(pady=10)

# Original selected image
original_label = tk.Label(root)
original_label.pack(pady=5)
tk.Label(root, text="Original Image").pack()

# Padded image (model input)
padded_label = tk.Label(root)
padded_label.pack(pady=5)
tk.Label(root, text="Padded Image (Model Input)").pack()

# Prediction result
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 16), wraplength=380)
result_label.pack(pady=10)

root.mainloop()