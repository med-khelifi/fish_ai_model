import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import json

# ------------------- إعدادات -------------------
IMG_SIZE = 224
TFLITE_FILE = "model.tflite"
CLASS_INDICES_FILE = "class_indices.json"
IMAGE_PATH = "test.jpg"   # غيّرها لصورتك

# ------------------- تحميل labels -------------------
with open(CLASS_INDICES_FILE, "r", encoding="utf-8") as f:
    class_map = json.load(f)

# ------------------- تحميل النموذج -------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------- دالة preprocessing -------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    
    # Resize مع الحفاظ على النسبة + padding
    img = ImageOps.pad(img, (IMG_SIZE, IMG_SIZE), method=Image.BILINEAR)
    
    img = np.array(img) / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)
    
    return img

# ------------------- تجهيز الصورة -------------------
img = preprocess_image(IMAGE_PATH)

# ------------------- prediction -------------------
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])

pred_index = int(np.argmax(output))
confidence = float(np.max(output))

# ------------------- النتيجة -------------------
# أحيانًا المفاتيح تكون string
pred_key = str(pred_index) if str(pred_index) in class_map else pred_index

result = class_map[pred_key]

print("\n🐟 النتيجة:")
print("ID:", result["fish_id"])
print("الاسم بالعربية:", result["name_ar"])
print("Nom français:", result["name_fr"])
print("Confidence:", confidence)

# ------------------- عرض كل الاحتمالات (اختياري) -------------------
print("\n📊 كل الاحتمالات:")
for i, prob in enumerate(output[0]):
    print(f"Class {i}: {prob:.4f}")