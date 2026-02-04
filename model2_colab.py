# Colab-ready training script for data2.npz on Google Drive
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Dense, Bidirectional, LSTM, Lambda, Permute,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===============================
# PATHS (Google Drive)
# ===============================
DATA_PATH = "/content/drive/MyDrive/htr/data2.npz"
BEST_MODEL_PATH = "/content/drive/MyDrive/htr/htr_ctc_words_multilang_best_v2.keras"
FINAL_MODEL_PATH = "/content/drive/MyDrive/htr/htr_ctc_words_multilang_final_v2.keras"

# ===============================
# LOAD PROCESSED DATA (pre2)
# ===============================
data = np.load(DATA_PATH, allow_pickle=True)
images = data["images"]
texts = data["labels"]

print(f"Loaded {len(images)} samples")

# ===============================
# BUILD MULTI-LANGUAGE CHARSET
# ===============================
charset = sorted(set("".join(texts)))
char_to_idx = {c: i for i, c in enumerate(charset)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

num_chars = len(charset)
blank_idx = num_chars  # REQUIRED FOR CTC

print("Total unique characters:", num_chars)

max_label_len = max(len(t) for t in texts)
print("Max label length:", max_label_len)

# ===============================
# CTC-SAFE LABEL ENCODING
# ===============================
labels = np.full(
    (len(texts), max_label_len),
    fill_value=blank_idx,
    dtype=np.int32
)

for i, text in enumerate(texts):
    for j, ch in enumerate(text):
        labels[i, j] = char_to_idx[ch]

label_length = np.array(
    [[len(t)] for t in texts],
    dtype=np.int32
)

# ===============================
# INPUT LENGTH (FROM MODEL SHAPE)
# ===============================
input_length = np.ones((len(images), 1), dtype=np.int32) * 64  # time steps after conv+pool

# ===============================
# DATA AUGMENTATION
# ===============================
augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomTranslation(0.02, 0.02),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="augment"
)

# ===============================
# MODEL DEFINITION
# ===============================
image_input = Input(shape=(32, 256, 1), name="image")

x = augment(image_input)

x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Permute((2, 1, 3))(x)
x = Reshape((64, 8 * 256))(x)

x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)

x = Dropout(0.2)(x)

y_pred = Dense(num_chars + 1, activation="softmax", name="y_pred")(x)

# ===============================
# CTC LOSS
# ===============================
labels_input = Input(shape=(max_label_len,), dtype="int32", name="labels")
input_len_input = Input(shape=(1,), dtype="int32", name="input_length")
label_len_input = Input(shape=(1,), dtype="int32", name="label_length")


def ctc_loss(args):
    y_pred, labels, input_len, label_len = args
    return tf.keras.backend.ctc_batch_cost(
        labels, y_pred, input_len, label_len
    )

loss = Lambda(ctc_loss, name="ctc")(
    [y_pred, labels_input, input_len_input, label_len_input]
)

model = Model(
    inputs=[image_input, labels_input, input_len_input, label_len_input],
    outputs=loss
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=lambda y_true, y_pred: y_pred
)

model.summary()

# ===============================
# CALLBACKS: BEST MODEL & EARLY STOP
# ===============================
checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

# ===============================
# TRAIN
# ===============================
history = model.fit(
    x=[images, labels, input_length, label_length],
    y=np.zeros(len(images)),  # dummy, not used
    batch_size=16,
    epochs=60,
    validation_split=0.1,      # 10% for validation
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ===============================
# SAVE FINAL MODEL
# ===============================
model.save(FINAL_MODEL_PATH)
print("Training complete, final model saved.")
print("Best model saved at:", BEST_MODEL_PATH)
