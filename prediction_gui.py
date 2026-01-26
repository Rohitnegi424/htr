import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Dense, Bidirectional, LSTM, Permute
from tensorflow.keras.models import Model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

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

# =============================== PREPROCESS IMAGE FOR WORD ===============================
def preprocess_word_image(img):
    """
    Preprocess a single word image to feed the model:
    - Resize to height=32
    - Pad width to max 256
    """
    IMG_HEIGHT = 32
    MAX_WIDTH = 256

    h, w = img.shape
    scale = IMG_HEIGHT / h
    new_w = int(w * scale)
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH

    img_resized = cv2.resize(img, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((IMG_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img_resized / 255.0

    model_input = np.expand_dims(padded, axis=-1)
    model_input = np.expand_dims(model_input, axis=0)
    return model_input

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

# =============================== SEGMENT PARAGRAPH INTO WORDS ===============================
def segment_paragraph(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation to join letters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours (words)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    word_imgs = []
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        word_img = img[y:y+h, x:x+w]
        word_imgs.append(word_img)
        boxes.append((x, y, w, h))

    # Sort words top-to-bottom, left-to-right
    boxes_words = sorted(zip(boxes, word_imgs), key=lambda b: (b[0][1], b[0][0]))
    sorted_word_imgs = [w for b, w in boxes_words]
    sorted_boxes = [b for b, w in boxes_words]

    return sorted_word_imgs, sorted_boxes

# =============================== PREDICT PARAGRAPH ===============================
def predict_paragraph():
    file_path = filedialog.askopenfilename(
        title="Select Paragraph Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        return

    try:
        word_imgs, boxes = segment_paragraph(file_path)
        predicted_words = []
        for w_img in word_imgs:
            model_input = preprocess_word_image(w_img)
            pred = model.predict(model_input)
            text = decode_prediction(pred)
            predicted_words.append(text)

        paragraph_text = " ".join(predicted_words)
        result_var.set(paragraph_text)

        # Display original image
        orig_img = Image.open(file_path)
        orig_img.thumbnail((500, 200))
        orig_img_tk = ImageTk.PhotoImage(orig_img)
        original_label.config(image=orig_img_tk)
        original_label.image = orig_img_tk

    except Exception as e:
        messagebox.showerror("Error", str(e))

# =============================== GUI ===============================
root = tk.Tk()
root.title("HTR Paragraph Prediction GUI")

tk.Button(root, text="Select Paragraph Image & Predict", command=predict_paragraph).pack(pady=10)

# Original paragraph image
original_label = tk.Label(root)
original_label.pack(pady=5)
tk.Label(root, text="Original Paragraph Image").pack()

# Prediction result
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 16), wraplength=480, justify="left")
result_label.pack(pady=10)

root.mainloop()