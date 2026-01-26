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

# IMPORTANT FIX: use model.inputs[0]
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
def preprocess_word(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = img.astype(np.float32) / 255.0
    h, w = img.shape

    scale = IMG_HEIGHT / h
    new_w = min(int(w * scale), MAX_WIDTH)

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    # BLACK padding (match training)
    padded = np.zeros((IMG_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img

    padded = np.expand_dims(padded, axis=-1)  # (32, 256, 1)
    padded = np.expand_dims(padded, axis=0)   # (1, 32, 256, 1)

    # dynamic time steps for CTC
    time_steps = max(1, new_w // 4)

    return padded, time_steps

# =========================================================
# CTC GREEDY DECODER
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
def predict_word(img_path):
    img, time_steps = preprocess_word(img_path)
    preds = infer_model.predict(img, verbose=0)

    seq = ctc_decode(preds, time_steps)

    text = ""
    for idx in seq:
        if idx == -1 or idx == blank_idx:
            continue
        text += idx_to_char[idx]

    return unicodedata.normalize("NFC", text)

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    test_image = r"D:\project\iam_words\words\a05\a05-053\a05-053-00-07.png"  # CHANGE THIS
    print("Prediction:", predict_word(test_image))
