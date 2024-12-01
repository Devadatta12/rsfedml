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

def load_images_as_numpy(data_dir, target_size=(256, 256)):
    images = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            img = cv2.resize(img, target_size)
            img = img / 255.0
            images.append(img)

    return np.array(images, dtype=np.float32)


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

train_images = load_images_as_numpy("../../../../data/clustering_dataset/train", target_size=(256, 256))
test_images = load_images_as_numpy("../../../../data/clustering_dataset/test", target_size=(256, 256))

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


pd.DataFrame(encoded_train).to_csv("encoded_features_train.csv", index=False)
pd.DataFrame(encoded_test).to_csv("encoded_features_test.csv", index=False)