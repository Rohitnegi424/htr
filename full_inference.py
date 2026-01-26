import cv2
import numpy as np
import tensorflow as tf
import unicodedata

# =========================================================
# CONFIG
# =========================================================
IMG_HEIGHT = 32
MAX_WIDTH = 256
MODEL_PATH = r"D:/project/htr_ctc_words_multilang_best.keras"
DATA_PATH = r"D:/project/processed_data/data.npz"

# =========================================================
# DUMMY CTC LOSS (REQUIRED FOR LOADING)
# =========================================================
def ctc_loss(args):
    return args[0]  # never used in inference

# =========================================================
# LOAD CHARSET (MUST MATCH TRAINING)
# =========================================================
data = np.load(DATA_PATH, allow_pickle=True)
texts = data["labels"]

charset = sorted(set("".join(texts)))
idx_to_char = {i: c for i, c in enumerate(charset)}
blank_idx = len(charset)

print("Charset size:", len(charset))

# =========================================================
# LOAD TRAINED MODEL
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
# WORD IMAGE PREPROCESSING
# =========================================================
def preprocess_word(img):
    if img is None:
        raise ValueError("Invalid image input")

    img = img.astype(np.float32) / 255.0
    h, w = img.shape

    scale = IMG_HEIGHT / h
    new_w = max(1, int(w * scale))  # avoid 0 width
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    # BLACK padding (match training)
    padded = np.zeros((IMG_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img

    padded = np.expand_dims(padded, axis=-1)  # (32, 256, 1)
    padded = np.expand_dims(padded, axis=0)   # (1, 32, 256, 1)

    time_steps = max(1, new_w // 4)
    return padded, time_steps

# =========================================================
# CTC DECODER
# =========================================================
def ctc_decode(preds, input_len):
    preds = preds[:, :input_len, :]
    decoded, _ = tf.keras.backend.ctc_decode(
        preds,
        input_length=[input_len],
        greedy=True
    )
    return decoded[0].numpy()[0]

# =========================================================
# WORD PREDICTION
# =========================================================
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
# LINE + WORD SEGMENTATION
# =========================================================
def segment_paragraph(paragraph_img):
    # Convert to binary image
    _, thresh = cv2.threshold(paragraph_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation for line segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find line contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # skip tiny noise
            line_img = paragraph_img[y:y+h, x:x+w]
            lines.append((y, line_img))

    # Sort lines top to bottom
    lines = sorted(lines, key=lambda x: x[0])
    lines = [l[1] for l in lines]

    # Now segment words in each line
    paragraph_words = []
    for line in lines:
        _, line_thresh = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Dilation for word segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_line = cv2.dilate(line_thresh, kernel, iterations=1)

        word_contours, _ = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        words = []
        for wc in word_contours:
            x, y, w, h = cv2.boundingRect(wc)
            if w > 5 and h > 5:  # skip tiny artifacts
                word_img = line[y:y+h, x:x+w]
                words.append((x, word_img))
        # Sort words left to right
        words = sorted(words, key=lambda x: x[0])
        words = [w[1] for w in words]
        paragraph_words.append(words)

    return paragraph_words

# =========================================================
# PREDICT PARAGRAPH
# =========================================================
def predict_paragraph(paragraph_img_path):
    paragraph_img = cv2.imread(paragraph_img_path, cv2.IMREAD_GRAYSCALE)
    if paragraph_img is None:
        raise ValueError(f"Cannot read paragraph image: {paragraph_img_path}")

    all_lines = segment_paragraph(paragraph_img)
    paragraph_text = ""
    for line_words in all_lines:
        line_text = " ".join([predict_word(w) for w in line_words])
        paragraph_text += line_text + "\n"

    return paragraph_text.strip()

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    paragraph_image = r"D:\project\test\a01-000u.png\a01-000u.png"  # CHANGE THIS
    print("Predicted paragraph:\n")
    print(predict_paragraph(paragraph_image))
