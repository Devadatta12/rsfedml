import os
from PIL import Image
import random

target_dir = "data/dataset"
train_dir = target_dir+"/train"
test_dir = target_dir+"/test"

os.makedirs(target_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def make_dataset(source_dir: str, split_ratio):
    for rs_category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, rs_category)
        if not os.path.isdir(category_path):
            continue

        for rs_class in os.listdir(category_path):
            class_path = os.path.join(category_path, rs_class)
            if not os.path.isdir(class_path):
                continue

            class_train_dir = os.path.join(train_dir, rs_class)
            class_test_dir = os.path.join(test_dir, rs_class)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            rs_images = [f for f in os.listdir(class_path) if f.endswith(".tif")]
            random.shuffle(rs_images)

            n_train = int(len(rs_images) * split_ratio)
            for i, image_name in enumerate(rs_images):
                src_img_path = os.path.join(class_path, image_name)
                tar_dir = class_train_dir if i < n_train else class_test_dir
                tar_img_name = os.path.splitext(image_name)[0] + ".png"
                tar_img_path = os.path.join(tar_dir, tar_img_name)

                try:
                    with Image.open(src_img_path) as img:
                        img = img.convert("RGB")
                        img.save(tar_img_path, format="PNG")
                except Exception as e:
                    print(f"Failed to process {src_img_path}: {e}")

make_dataset("data/RSI-CB256", 0.8)