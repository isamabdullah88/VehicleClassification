import keras.layers as L
from config import *
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.optimizers import Adam


def build_model(custom_model=True):
    """
    Builds a custom model based on required task.
    """
    if custom_model:
        return custom_model()

    return transfer_learn_model()


def transfer_learn_model():
    base_model = VGG16(weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)

    x = base_model.output
    # x = L.MaxPooling2D()(x)
    x = L.Flatten()(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    for layer in model.layers[:21]:
        layer.trainable = False

    optim = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model


def custom_model():
    model = Sequential()
    model.add(L.Conv2D(32, kernel_size=3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(256, kernel_size=3, activation='relu'))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.Flatten())
    model.add(L.BatchNormalization())
    model.add(L.Dense(512, activation='relu'))
    model.add(L.BatchNormalization())
    model.add(L.Dense(NUM_CLASSES, activation='softmax'))

    optim = Adam()
    # optim = SGD(lr=1e-4, momentum=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model
