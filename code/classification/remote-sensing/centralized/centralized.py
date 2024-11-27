import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD


train_dir = "../../../../data/testing/train"

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    # validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# val_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode="categorical",
#     subset="validation"
# )

test_dir = "../../../../data/testing/test"
test_datagen = ImageDataGenerator(
    rescale=1.0/255.0   
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu", kernel_regularizer=regularizers.L2(0.003)),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

optimizer = SGD(learning_rate=0.03)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    validation_data = test_generator,
    epochs=10
)

testloss, testaccuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {testaccuracy:.4f}")