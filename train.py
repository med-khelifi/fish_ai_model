import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ----------------------------- إعدادات -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10                  # يمكن زيادتها لنتائج أفضل
CONFIG_FILE = "species_config.json"
DATASET_DIR = "dataset"
MODEL_H5 = "fish_model.h5"
TFLITE_FILE = "model.tflite"
LABELS_FILE = "labels.txt"
CLASS_INDICES_FILE = "class_indices.json"

# ----------------------------- 1. تحميل التكوين -----------------------------
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    species_list = json.load(f)

# بناء قاموس: اسم_المجلد -> معرف السمكة
folder_to_id = {s["folder"]: s["id"] for s in species_list}
print(f"✅ تم تحميل {len(species_list)} نوع سمكة من {CONFIG_FILE}")

# ----------------------------- 2. محسن البيانات -----------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,        # 20% للتحقق
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# عدد الأصناف (يجب أن يساوي عدد مجلداتك)
num_classes = len(train_generator.class_indices)
print(f"📊 عدد الأصناف المكتشفة: {num_classes}")

# ----------------------------- 3. بناء النموذج -----------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# تجميد طبقات القاعدة
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------- 4. التدريب -----------------------------
print("\n🚀 بدء التدريب...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# حفظ النموذج الكامل (اختياري، للنسخ الاحتياطي)
model.save(MODEL_H5)
print(f"💾 تم حفظ النموذج H5: {MODEL_H5}")

# ----------------------------- 5. تصدير TFLite -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)
print(f"📱 تم تصدير TFLite: {TFLITE_FILE}")

# ----------------------------- 6. بناء class_indices.json (ربط الفهرس بمعرف السمكة) -----------------------------
# train_generator.class_indices: { "001_الوراطة": 0, "002_القاروص": 1, ... }
class_indices_mapping = {}
for folder_name, class_idx in train_generator.class_indices.items():
    fish_id = folder_to_id.get(folder_name, None)
    if fish_id is None:
        print(f"⚠️ تحذير: المجلد {folder_name} غير موجود في التكوين! سيتم تجاهله.")
        continue
    class_indices_mapping[class_idx] = {
        "fish_id": fish_id,
        "name_ar": next((s["name_ar"] for s in species_list if s["folder"] == folder_name), ""),
        "name_fr": next((s["name_fr"] for s in species_list if s["folder"] == folder_name), "")
    }

# حفظ الملف
with open(CLASS_INDICES_FILE, "w", encoding="utf-8") as f:
    json.dump(class_indices_mapping, f, ensure_ascii=False, indent=2)
print(f"🔗 تم حفظ ربط الفئات: {CLASS_INDICES_FILE}")

# ----------------------------- 7. كتابة labels.txt (للتوثيق فقط) -----------------------------
with open(LABELS_FILE, "w", encoding="utf-8") as f:
    for folder_name in sorted(train_generator.class_indices, key=lambda k: train_generator.class_indices[k]):
        f.write(f"{folder_name}\n")
print(f"🏷️  تم حفظ: {LABELS_FILE}")

print("\n✅ تم إكمال التدريب والتصدير بنجاح!")