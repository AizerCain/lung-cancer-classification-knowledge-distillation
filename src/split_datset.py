import os
import shutil
import random
from tqdm import tqdm

# -------------------------
# PATHS (YOUR SYSTEM)
# -------------------------
BASE_DIR = r"D:\DL Project\lung_cancer_project"
SOURCE_DIR = os.path.join(BASE_DIR, "lung_image_sets")

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_split")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR   = os.path.join(OUTPUT_DIR, "val")
TEST_DIR  = os.path.join(OUTPUT_DIR, "test")

# -------------------------
# SPLIT RATIOS
# -------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# -------------------------
# CREATE OUTPUT DIRS
# -------------------------
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# SPLIT FUNCTION
# -------------------------
def split_class(class_name):
    src_class_path = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(src_class_path)

    random.shuffle(images)
    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end   = train_end + int(total * VAL_RATIO)

    splits = {
        TRAIN_DIR: images[:train_end],
        VAL_DIR:   images[train_end:val_end],
        TEST_DIR:  images[val_end:]
    }

    for split_dir, img_list in splits.items():
        dst_class_dir = os.path.join(split_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        for img in tqdm(img_list, desc=f"{class_name} → {os.path.basename(split_dir)}"):
            src = os.path.join(src_class_path, img)
            dst = os.path.join(dst_class_dir, img)
            shutil.copy2(src, dst)

# -------------------------
# RUN SPLIT
# -------------------------
classes = os.listdir(SOURCE_DIR)

for cls in classes:
    split_class(cls)

print("\n✅ Dataset split completed successfully!")
