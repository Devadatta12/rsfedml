"""test: A Flower / TensorFlow app."""

import os
import numpy as np
import keras
from keras import layers, models, regularizers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import load_dataset
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import cv2

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def resize_images(images, target_size=(224, 224)):
    print("Shape of the images is ",images.shape)
    batch_images = tf.convert_to_tensor(images, dtype=tf.float32)
    resized_images = tf.image.resize(batch_images, target_size)
    resized_images = resized_images / 255.0

    return resized_images

def load_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.L2(0.003)),
        layers.Dense(35, activation="softmax")
    ])

    optimizer = SGD(learning_rate=0.03)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    global partitioner
    if fds is None:
        dataset_dict = load_dataset("imagefolder", data_dir="../../../../data/dataset")
        fds = dataset_dict["train"]
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = fds
    partition = partitioner.load_partition(partition_id)
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    train_images = [cv2.resize(img, (224, 224)) for img in partition["train"]["image"]]
    test_images = [cv2.resize(img, (224, 224)) for img in partition["test"]["image"]]

#     for i, img in enumerate(partition["train"]["image"][:5]):
#         print(f"Image {i} type: {type(img)}")
#         print(f"Image {i} shape: {np.array(img).shape if isinstance(img, np.ndarray) else 'Unknown'}")

# # Check for irregularities in shapes
#     shapes = [np.array(img).shape for img in partition["train"]["image"]]
#     print("Unique shapes in train images:", set(shapes))
    x_train, y_train = np.array(train_images, dtype=np.float32) / 255.0, partition["train"]["label"]
    x_test, y_test = np.array(test_images, dtype=np.float32) / 255.0, partition["test"]["label"]
    return x_train, y_train, x_test, y_test
