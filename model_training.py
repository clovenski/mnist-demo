#!/usr/bin/python

from __future__ import print_function
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# input to model needs to be of shape (N, 28, 28, 1) where N is number of samples
x_train, x_test = x_train[:,:,:,np.newaxis], x_test[:,:,:,np.newaxis]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 2, strides=2, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 2, strides=1, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
eval = model.evaluate(x_test, y_test)
print(eval)
model.save('model.h5')