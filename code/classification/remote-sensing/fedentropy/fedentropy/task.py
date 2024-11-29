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
from scipy.ndimage import affine_transform
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

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

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


fds = None  # Cache FederatedDataset


def entropy(class_distribution: dict):
    total_samples = sum(class_distribution.values())
    probabilities = [count / total_samples for count in class_distribution.values()]
    dataset_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return dataset_entropy


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    training_path = "../../../../data/fedarated_dataset/iid/train/partition_"+str(partition_id)
    testing_path = "../../../../data/fedarated_dataset/iid/test/partition_"+str(partition_id)
    
    train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
    )


    train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    test_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.1,
    zoom_range=0.1   
    )

    test_generator = test_datagen.flow_from_directory(
    testing_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
    )

    class_distribution = Counter(train_generator.classes)
    dataset_entropy = entropy(class_distribution)

    return train_generator, test_generator, dataset_entropy
