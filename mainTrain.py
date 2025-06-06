import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory = '/Users/jalajbalodi/Desktop/BrainTumor Classification DL/datasets'

no_tumor_images = os.listdir(image_directory + '/no')
yes_tumor_images = os.listdir(image_directory + '/yes')

dataset = []
label = []

INPUT_SIZE = 64

for image_name in no_tumor_images:
    if image_name.endswith('.jpg'):
        image_path = image_directory + '/no/' + image_name
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Couldn't read image {image_path}")
            continue
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # convert BGR to RGB
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.endswith('.jpg'):
        image_path = image_directory + '/yes/' + image_name
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Couldn't read image {image_path}")
            continue
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # convert BGR to RGB
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Model Building
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=16,
          verbose=1,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=False)

model.save('/Users/jalajbalodi/Desktop/BrainTumor Classification DL/BrainTumor10EpochsCategorical.h5')
