import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape, Dense,
    BatchNormalization, Bidirectional, LSTM, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# PATH SETTINGS
# -----------------------------
project_dir = r"D:/project"
data_path = os.path.join(project_dir, "processed_data/data.npz")

# -----------------------------
# LOAD DATA
# -----------------------------
data = np.load(data_path, allow_pickle=True)
images = data["images"]
labels = data["labels"]
print(f"✅ Loaded {len(images)} images")

# -----------------------------
# CHARACTER SET + ENCODING
# -----------------------------
char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
num_classes = len(char_set) + 1  # +1 for CTC blank
char_to_int = {c: i + 1 for i, c in enumerate(char_set)}  # 1..62
int_to_char = {i + 1: c for i, c in enumerate(char_set)}

def encode_text(text):
    return [char_to_int[c] for c in text if c in char_to_int]

encoded_labels = [encode_text(t) for t in labels]

# -----------------------------
# FILTER EMPTY LABELS
# -----------------------------
filtered_images = []
filtered_labels = []

for img, lbl in zip(images, encoded_labels):
    if len(lbl) > 0:
        filtered_images.append(img)
        filtered_labels.append(lbl)

images = np.array(filtered_images)
encoded_labels = filtered_labels
print(f"✅ Filtered images with non-empty labels: {len(images)}")

max_label_len = max(len(l) for l in encoded_labels)
print(f"Max label length: {max_label_len}")

# -----------------------------
# CLIP LABELS & PAD FOR CTC
# -----------------------------
# Clip labels to max valid index for CTC: num_classes-2
encoded_labels = [[min(x, num_classes - 2) for x in l] for l in encoded_labels]

padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
    encoded_labels, maxlen=max_label_len, padding='post', value=0
)

# -----------------------------
# VALIDATE LABELS
# -----------------------------
for i, l in enumerate(padded_labels):
    if np.any(l >= num_classes - 1):
        raise ValueError(f"Invalid label in index {i}: {l}")

print("✅ All labels validated successfully")

# -----------------------------
# CRNN + CTC MODEL
# -----------------------------
input_img = Input(shape=(32, 128, 1), name="image_input")
labels_input = Input(shape=(max_label_len,), dtype="int32", name="labels")
input_length = Input(shape=(1,), dtype="int32", name="input_length")
label_length = Input(shape=(1,), dtype="int32", name="label_length")

x = Conv2D(64, (3,3), activation="relu", padding="same")(input_img)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

x = Reshape(target_shape=(-1, 128))(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)

y_pred = Dense(num_classes, activation="softmax", name="y_pred")(x)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
    [y_pred, labels_input, input_length, label_length]
)

model = Model(
    inputs=[input_img, labels_input, input_length, label_length],
    outputs=loss_out
)
model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer="adam")

# -----------------------------
# TRAIN DATA PREP
# -----------------------------
input_len = np.ones((len(images), 1), dtype=np.int32) * (images.shape[2] // 4)
label_len = np.array([[len(l)] for l in encoded_labels], dtype=np.int32)

# -----------------------------
# TRAINING SETUP
# -----------------------------
best_model_path = os.path.join(project_dir, "handwriting_best.keras")
final_model_path = os.path.join(project_dir, "handwriting_final.keras")

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

# -----------------------------
# TRAINING
# -----------------------------
history = model.fit(
    x=[images, padded_labels, input_len, label_len],
    y=np.zeros(len(images)),  # dummy for CTC
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save(final_model_path)
print("✅ Model trained and saved successfully!")