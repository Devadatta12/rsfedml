import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape, Conv2DTranspose, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import pandas as pd


import os
import numpy as np
import cv2

import os
import numpy as np
import cv2  # For image resizing

def load_images_and_labels(data_dir, target_size=(256, 256)):
    images = []
    labels = []
    class_map = {}

    for label, classname in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, classname)
        if not os.path.isdir(class_dir):
            continue
        class_map[classname] = label
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(file_path)
                img = cv2.resize(img, target_size)
                img = img / 255.0
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels), class_map


def build_autoencoder(input_shape=(256, 256, 3), latent_dim=128):
    encoder_input = Input(shape=input_shape)
    resnet_encoder = ResNet50(weights=None, include_top=False, input_tensor=encoder_input)
    encoded = GlobalAveragePooling2D()(resnet_encoder.output)
    latent_space = Dense(latent_dim, activation='relu', name="latent_space", kernel_regularizer=l2(0.003))(encoded)

    decoder_input = Dense(8 * 8 * 512, activation='relu', kernel_regularizer=l2(0.003))(latent_space)
    decoder_reshape = Reshape((8, 8, 512))(decoder_input)
    decoder = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.003))(decoder_reshape)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.003))(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.003))(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=l2(0.003))(decoder)

    autoencoder = Model(inputs=encoder_input, outputs=decoder_output)

    encoder = Model(inputs=encoder_input, outputs=latent_space)

    return autoencoder, encoder


def save_encoded_features_with_labels(encoded_features, labels, output_file):
    df = pd.DataFrame(encoded_features)
    df['label'] = labels

    df.to_csv(output_file, index=False)


train_images, train_labels, class_map = load_images_and_labels("../../../../data/dataset/train", target_size=(256, 256))
test_images, test_labels, _ = load_images_and_labels("../../../../data/dataset/test", target_size=(256, 256))


autoencoder, encoder = build_autoencoder(latent_dim=128)

sgd_optimizer = SGD(learning_rate=0.03)
autoencoder.compile(optimizer='sgd', loss='mse')

autoencoder.fit(
    train_images,
    train_images,
    validation_data=(test_images, test_images),
    epochs=10,
    batch_size=32
)

encoded_train = encoder.predict(train_images)
encoded_test = encoder.predict(test_images)

train_df = pd.DataFrame(encoded_train)
train_df['label'] = train_labels

test_df = pd.DataFrame(encoded_test)
test_df['label'] = test_labels

all_df = pd.concat([train_df, test_df], ignore_index=True)

all_df.to_csv("encoded_features.csv", index=False)
