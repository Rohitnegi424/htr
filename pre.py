import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# PATH SETTINGS
# -----------------------------
images_dir = r"D:/project/iam_words/words"
labels_file = r"D:/project/iam_words/words.txt"
output_dir = r"D:/project/processed_data"
failed_images_log = r"D:/project/failed_images.txt"

os.makedirs(output_dir, exist_ok=True)

print("Starting preprocessing...")

# -----------------------------
# READ LABELS FROM words.txt
# -----------------------------
labels = {}
with open(labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.startswith("#") and line.strip():
            parts = line.strip().split()
            if len(parts) >= 9:
                image_id = parts[0]
                label = " ".join(parts[8:]).strip()  # transcription

                # ✅ Skip empty or unreadable labels
                if label == "" or label == " ":
                    continue

                labels[image_id] = label

print(f"✅ Total valid labels read: {len(labels)}")

# -----------------------------
# PREPROCESS IMAGES
# -----------------------------
images = []
labels_list = []
processed = 0

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05
)

for subdir, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(subdir, file)
            image_id = os.path.splitext(file)[0]

            if image_id not in labels:
                continue

            label = labels[image_id]
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                with open(failed_images_log, "a") as logf:
                    logf.write(f"Failed to load {file_path}\n")
                continue

            # Resize and normalize
            img = cv2.resize(img, (128, 32))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)

            # Add augmented versions
            aug_iter = datagen.flow(np.expand_dims(img, 0), batch_size=1)
            for _ in range(3):  # 3 augmented samples
                aug_img = next(aug_iter)[0]
                images.append(aug_img)
                labels_list.append(label)

            # Add original
            images.append(img)
            labels_list.append(label)

            processed += 1
            if processed % 200 == 0:
                print(f"Processed {processed} images...")

print(f"✅ Total processed images: {processed}")

# -----------------------------
# SAVE DATA
# -----------------------------
images_np = np.array(images)
labels_np = np.array(labels_list, dtype=object)

# ✅ Filter out any remaining empty labels
valid_idx = [i for i, lbl in enumerate(labels_np) if lbl.strip() != ""]
images_np = images_np[valid_idx]
labels_np = labels_np[valid_idx]

np.savez_compressed(os.path.join(output_dir, "data.npz"),
                    images=images_np, labels=labels_np)

print("✅ Data saved to:", os.path.join(output_dir, "data.npz"))
print(f"✅ Final image count: {len(images_np)}")