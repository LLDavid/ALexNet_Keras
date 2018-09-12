import tensorflow as tf
from tensorflow import keras
import numpy as np

## Load data: cifar10
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()

no_classes=10

## Build NN
model=keras.Sequential()
## 1st layer: 11*11*3, #=96, stride=4
model.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding="same",
                              activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

## 2nd layer: 3*3*96, #=256, stride=1
model.add(keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=1, padding="same",
                              activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

## 3rd layer: 3*3*256, #=384, stide=1 (no pooling layer)
model.add(keras.layers.Conv2D(filters=394, kernel_size=(3,3), strides=1, padding="same",
                              activation="relu"))

## 4th layer: 3*3*384, #=384, stide=1 (no pooling layer)
model.add(keras.layers.Conv2D(filters=394, kernel_size=(3,3), strides=1, padding="same",
                              activation="relu"))

## 5th layer: 3*3*384, #=256, stide=1
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same",
                              activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

## 6th layer: Dense layer
model.add(keras.layers.Dense(4096, activation='relu', use_bias=True))

## 7th layer: Dense layer
model.add(keras.layers.Dense(4096, activation='relu', use_bias=True))

## 8th layer: Output layer (Dense)
model.add(keras.layers.Dense(1, activation='softmax'))


model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=50)