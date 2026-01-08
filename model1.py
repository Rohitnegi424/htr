import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Dense, Bidirectional, LSTM, Lambda, Permute
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================
# LOAD PROCESSED DATA
# ===============================
data = np.load(r"D:/project/processed_data/data.npz", allow_pickle=True)
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
# MODEL DEFINITION
# ===============================
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
    optimizer="adam",
    loss=lambda y_true, y_pred: y_pred
)

model.summary()

# ===============================
# CALLBACKS: BEST MODEL & EARLY STOP
# ===============================
best_model_path = r"D:/project/htr_ctc_words_multilang_best.keras"

checkpoint = ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ===============================
# TRAIN
# ===============================
history = model.fit(
    x=[images, labels, input_length, label_length],
    y=np.zeros(len(images)),  # dummy, not used
    batch_size=16,
    epochs=50,
    validation_split=0.1,      # 10% for validation
    callbacks=[checkpoint, early_stop]
)

# ===============================
# SAVE FINAL MODEL
# ===============================
final_model_path = r"D:/project/htr_ctc_words_multilang_final.keras"
model.save(final_model_path)
print("Training complete, final model saved.")
print("Best model saved at:", best_model_path)