import os
import shutil
import numpy as np
import cv2
from collections import defaultdict

def resize_and_copy_images(source_dir, target_dir, target_size=(224, 224)):
    os.makedirs(target_dir, exist_ok=True)
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        target_class_path = os.path.join(target_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, target_size)
                target_image_path = os.path.join(target_class_path, image_name)
                cv2.imwrite(target_image_path, resized_image)

def create_iid_partitions(source_dir, output_dir, num_partitions, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    
    class_dirs = [os.path.join(source_dir, cls) for cls in os.listdir(source_dir)]
    for partition_idx in range(num_partitions):
        partition_dir = os.path.join(output_dir, f"partition_{partition_idx}")
        os.makedirs(partition_dir, exist_ok=True)

    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        np.random.shuffle(images)
        partition_size = len(images) // num_partitions

        for i in range(num_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i != num_partitions - 1 else len(images)
            partition_images = images[start:end]

            for image_name in partition_images:
                src_image_path = os.path.join(class_dir, image_name)
                target_class_dir = os.path.join(output_dir, f"partition_{i}", class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                image = cv2.imread(src_image_path)
                resized_image = cv2.resize(image, target_size)
                target_image_path = os.path.join(target_class_dir, image_name)
                cv2.imwrite(target_image_path, resized_image)

def create_non_iid_partitions(source_dir, output_dir, num_partitions, classes_per_partition, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    class_dirs = [os.path.join(source_dir, cls) for cls in os.listdir(source_dir)]
    np.random.shuffle(class_dirs)

    for partition_idx in range(num_partitions):
        partition_dir = os.path.join(output_dir, f"partition_{partition_idx}")
        os.makedirs(partition_dir, exist_ok=True)

        start = partition_idx * classes_per_partition
        end = start + classes_per_partition
        assigned_classes = class_dirs[start:end]

        for class_dir in assigned_classes:
            class_name = os.path.basename(class_dir)
            target_class_dir = os.path.join(partition_dir, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            if not os.path.isdir(class_dir):
                continue

            for image_name in os.listdir(class_dir):
                if not image_name.lower().endswith('.png'):
                    continue
                src_image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(src_image_path)
                resized_image = cv2.resize(image, target_size)
                target_image_path = os.path.join(target_class_dir, image_name)
                cv2.imwrite(target_image_path, resized_image)

def main():
    dataset_dir = "data/dataset"
    output_iid_dir = "data/fedarated_dataset/iid"
    output_non_iid_dir = "data/fedarated_dataset/non_iid"
    num_partitions = 20
    classes_per_partition = 4

    # Process train and test sets for I.I.D.
    # create_iid_partitions(os.path.join(dataset_dir, "train"), os.path.join(output_iid_dir, "train"), num_partitions)
    # create_iid_partitions(os.path.join(dataset_dir, "test"), os.path.join(output_iid_dir, "test"), num_partitions)

    # Process train and test sets for non-I.I.D.
    create_non_iid_partitions(os.path.join(dataset_dir, "train"), os.path.join(output_non_iid_dir, "train"), num_partitions, classes_per_partition)
    create_non_iid_partitions(os.path.join(dataset_dir, "test"), os.path.join(output_non_iid_dir, "test"), num_partitions, classes_per_partition)

if __name__ == "__main__":
    main()
