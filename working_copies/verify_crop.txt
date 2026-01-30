import cv2
import numpy as np
import tensorflow as tf
import unicodedata
import os

# =========================================================
# CONFIG
# =========================================================
IMG_HEIGHT = 32
MAX_WIDTH = 256
MODEL_PATH = r"D:/project/htr_ctc_words_multilang_best.keras"
DATA_PATH = r"D:/project/processed_data/data.npz"
CROPPED_WORDS_DIR = r"D:/project/cropped_words"  # folder to save cropped word images

os.makedirs(CROPPED_WORDS_DIR, exist_ok=True)

# =========================================================
# DUMMY CTC LOSS (REQUIRED FOR LOADING)
# =========================================================
def ctc_loss(args):
    return args[0]

# =========================================================
# LOAD CHARSET
# =========================================================
data = np.load(DATA_PATH, allow_pickle=True)
texts = data["labels"]

charset = sorted(set("".join(texts)))
idx_to_char = {i: c for i, c in enumerate(charset)}
blank_idx = len(charset)

print("Charset size:", len(charset))

# =========================================================
# LOAD MODEL
# =========================================================
full_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"ctc_loss": ctc_loss},
    compile=False
)

image_input = full_model.inputs[0]
y_pred = full_model.get_layer("y_pred").output

infer_model = tf.keras.models.Model(
    inputs=image_input,
    outputs=y_pred
)

print("Inference model loaded successfully.")

# =========================================================
# PREPROCESS WORD IMAGE
# =========================================================
def preprocess_word(img):
    if img is None:
        raise ValueError("Invalid image")

    img = img.astype(np.float32) / 255.0
    h, w = img.shape
    scale = IMG_HEIGHT / h
    new_w = max(1, min(int(w * scale), MAX_WIDTH))
    img = cv2.resize(img, (new_w, IMG_HEIGHT))
    padded = np.zeros((IMG_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img
    padded = np.expand_dims(padded, axis=-1)
    padded = np.expand_dims(padded, axis=0)
    time_steps = max(1, new_w // 4)
    return padded, time_steps

# =========================================================
# CTC DECODER
# =========================================================
def ctc_decode(preds, input_len):
    preds = preds[:, :input_len, :]
    decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=[input_len], greedy=True)
    return decoded[0].numpy()[0]

def predict_word(img):
    img_prep, time_steps = preprocess_word(img)
    preds = infer_model.predict(img_prep, verbose=0)
    seq = ctc_decode(preds, time_steps)

    text = ""
    for idx in seq:
        if idx == -1 or idx == blank_idx:
            continue
        text += idx_to_char[idx]

    return unicodedata.normalize("NFC", text)

# =========================================================
# SEGMENT WORDS
# =========================================================
def segment_words(paragraph_img):
    gray = cv2.cvtColor(paragraph_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_imgs = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            word_img = gray[y:y+h, x:x+w]
            word_imgs.append(word_img)
            boxes.append((x, y, w, h))

    boxes, word_imgs = zip(*sorted(zip(boxes, word_imgs), key=lambda b: (b[0][1], b[0][0])))
    return word_imgs, boxes

# =========================================================
# PREDICT PARAGRAPH
# =========================================================
def predict_paragraph(paragraph_img, save_crops=True):
    words, boxes = segment_words(paragraph_img)
    predictions = []

    for i, word in enumerate(words):
        pred = predict_word(word)
        predictions.append(pred)

        if save_crops:
            crop_path = os.path.join(CROPPED_WORDS_DIR, f"word_{i}_{pred}.png")
            cv2.imwrite(crop_path, word)

    return " ".join(predictions)

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    paragraph_image = cv2.imread(r"D:\project\test\a01-000u.png\a01-000u.png")
    text = predict_paragraph(paragraph_image, save_crops=True)
    print("Predicted paragraph:")
    print(text)
    print(f"Cropped words saved in: {CROPPED_WORDS_DIR}")