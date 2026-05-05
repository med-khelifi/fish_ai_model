import json
import os

CONFIG_FILE = "species_config.json"
DATASET_DIR = "dataset"

# تحميل ملف التكوين
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    species_list = json.load(f)

# إنشاء مجلد dataset الرئيسي إذا لم يكن موجوداً
os.makedirs(DATASET_DIR, exist_ok=True)

created = 0
for species in species_list:
    folder_name = species["folder"]  # مثلاً "001_الوراطة"
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ تم إنشاء: {folder_path}")
        created += 1
    else:
        print(f"⏭️  موجود مسبقاً: {folder_path}")

print(f"\n🎉 تم تجهيز {created} مجلد جديد (إجمالي {len(species_list)} مجلد).")