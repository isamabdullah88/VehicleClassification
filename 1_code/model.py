import keras.layers as L
from config import *
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.optimizers import Adam

from resnet152 import resnet152_model


def build_model(model_type=None):
    """
    Builds a custom model based on required task.
    ARGS:
    model_type: Type of model to build.
    'custom_model': For custom layered model.
    'VGG16': For using VGG16 as base model.
    Default: Use resnet152 as base model.
    """
    if model_type == 'custom_model':
        return build_custom_model()
    elif model_type == 'VGG16':
        return build_VGG_model()
    else:
        return build_resnet152_model()


def build_resnet152_model():
    """
    Builds resnet152 model. Code partially used from another gist.
    """
    return resnet152_model(IMG_SIZE, IMG_SIZE, 3, NUM_CLASSES)


def build_VGG_model():
    """
    Builds VGG16 as base model.
    """
    base_model = VGG16(weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)

    x = L.Flatten()(base_model.output)
    x = L.BatchNormalization()(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    for layer in model.layers[:100]:
        layer.trainable = False

    optim = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optim)

    return model


def build_custom_model():
    """
    Builds a custom model.
    """
    model = Sequential()
    model.add(L.Conv2D(32, kernel_size=3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(L.Dropout(0.5))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(L.Dropout(0.5))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(L.Dropout(0.5))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(256, kernel_size=3, activation='relu'))
    model.add(L.Dropout(0.5))
    model.add(L.MaxPooling2D(pool_size=2))
    model.add(L.Flatten())
    model.add(L.BatchNormalization())
    model.add(L.Dense(512, activation='relu'))
    model.add(L.Dropout(0.5))
    model.add(L.BatchNormalization())
    model.add(L.Dense(NUM_CLASSES, activation='softmax'))

    optim = Adam()
    # optim = SGD(lr=1e-4, momentum=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model
