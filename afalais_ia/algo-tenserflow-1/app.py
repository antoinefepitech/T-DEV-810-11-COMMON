from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import sys
import time
import datetime
import shutil
import os
from PIL import Image
from shutil import rmtree


TRAINNING_RAW = "./raw/test/"
VALIDATION_RAW = "./raw/val/"
WORK_DIR = "./tmp/"
TRAINNING_DIR = "./tmp/test/"
VALIDATION_DIR = "./tmp/val/"
EPOCHS = 2


def init_dataset_files():
    print("start init")
    rmtree(WORK_DIR)
    os.mkdir(WORK_DIR)
    os.mkdir(TRAINNING_DIR)
    os.mkdir(TRAINNING_DIR+'NORMAL')
    os.mkdir(TRAINNING_DIR+'PNEUMONIA')
    os.mkdir(VALIDATION_DIR)
    os.mkdir(VALIDATION_DIR+'NORMAL')
    os.mkdir(VALIDATION_DIR+'PNEUMONIA')

    # resize(TRAINNING_RAW, 'NORMAL')
    resize(VALIDATION_RAW, 'PNEUMONIA', VALIDATION_DIR)
    resize(TRAINNING_RAW, 'PNEUMONIA', TRAINNING_DIR)
    resize(VALIDATION_RAW, 'NORMAL', VALIDATION_DIR)
    resize(TRAINNING_RAW, 'NORMAL', TRAINNING_DIR)


def resize(raw_path, typeImg, target, width=200, height=200):
    dirs = os.listdir(raw_path + typeImg)
    for item in dirs:
        path = raw_path + typeImg + '/' + item
        if os.path.isfile(path):
            im = Image.open(path)
            f = os.path.basename(path)
            imResize = im.resize((200, 200), Image.ANTIALIAS)
            imResize.save(target + typeImg +
                          '/' + f, 'JPEG', quality=90)


def init_generator():
    tr_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    tr_generator = tr_datagen.flow_from_directory(
        TRAINNING_DIR,
        target_size=(200, 200),
        class_mode='categorical')

    vaL_generator = tr_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(200, 200),
        class_mode='categorical')

    return tr_datagen, val_datagen, tr_generator, vaL_generator


init_dataset_files()
training_datagen, valid_datagen, training_generator, validation_generator = init_generator()

# verbose data
num_normal_tr = len(os.listdir(TRAINNING_DIR + 'NORMAL'))
num_pneumonia_tr = len(os.listdir(TRAINNING_DIR + 'PNEUMONIA'))
num_normal_val = len(os.listdir(VALIDATION_DIR + 'NORMAL'))
num_pneumonia_val = len(os.listdir(VALIDATION_DIR + 'PNEUMONIA'))
total_tr = num_normal_tr + num_pneumonia_tr
total_val = num_normal_val + num_pneumonia_val
print(total_tr)
print(total_val)
# verbose data

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

# model.fit_generator depreacted | todo: remplace model.fit_generator by model.fit
history = model.fit_generator(
    training_generator, validation_data=validation_generator, steps_per_epoch=total_tr, epochs=2, verbose=1)
