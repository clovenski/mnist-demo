#!/usr/bin/python

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# input to model needs to be of shape (N, 28, 28, 1) where N is number of samples
x_train, x_test = x_train[:,:,:,np.newaxis], x_test[:,:,:,np.newaxis]

train_idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=85,
    shear_range=15.0,
    fill_mode='constant',
    cval=0)

val_idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=75,
    shear_range=10.0,
    fill_mode='constant',
    cval=0)

train_dg = train_idg.flow(x_train, y_train)
train_steps = len(x_train) / train_dg.batch_size

val_dg = val_idg.flow(x_test, y_test)
val_steps = len(x_test) / val_dg.batch_size

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 2, strides=2, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 2, strides=1, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 2, strides=1, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 2, strides=1, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

rlrop = ReduceLROnPlateau(monitor='val_loss', patience=2)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit_generator(
    train_dg,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=val_dg,
    validation_steps=val_steps,
    callbacks=[rlrop, early_stop])
print(model.evaluate(x_test, y_test))
model.save('model.h5')