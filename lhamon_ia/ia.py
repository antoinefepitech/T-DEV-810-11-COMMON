import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


# Set up the folders
PATH = "./chest_x_ray/"

train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'val')

train_normal_dir = os.path.join(train_dir, 'NORMAL') 
train_pneumo_dir = os.path.join(train_dir, 'PNEUMONIA')
val_normal_dir = os.path.join(val_dir, 'NORMAL')
val_pneumo_dir = os.path.join(val_dir, 'PNEUMONIA')

# Data summary
number_train_normal = len(os.listdir(train_normal_dir))
number_train_pneumonia = len(os.listdir(train_pneumo_dir))
number_val_normal = len(os.listdir(val_normal_dir))
number_val_pneumo = len(os.listdir(val_pneumo_dir))

total_train = number_train_normal + number_train_pneumonia
total_val = number_val_normal + number_val_pneumo

print('Train normal images :')
print(number_train_normal)
print('Train pneumonia images :')
print(number_train_pneumonia)
print('Val normal images :')
print(number_val_normal)
print('Val pneumonia images :')
print(number_val_pneumo)

# Variables
batch_size = 32
epochs = 100
IMG_HEIGHT = 96
IMG_WIDTH = 96

METRICS = [
  tf.keras.metrics.BinaryAccuracy(name='accuracy', dtype=tf.float32),
  tf.keras.metrics.TruePositives(name='true_positives', dtype=tf.float32),
  tf.keras.metrics.FalsePositives(name='false_positives', dtype=tf.float32),
  tf.keras.metrics.TrueNegatives(name='true_negatives', dtype=tf.float32),
  tf.keras.metrics.FalseNegatives(name='false_negatives', dtype=tf.float32), 
  tf.keras.metrics.Precision(name='precision', dtype=tf.float32),
  tf.keras.metrics.Recall(name='recall', dtype=tf.float32),
  tf.keras.metrics.AUC(name='auc', dtype=tf.float32),
]

train_image_generator = ImageDataGenerator(rescale=1./255) 
val_image_generator = ImageDataGenerator(rescale=1./255) 

train_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_gen = val_image_generator.flow_from_directory(batch_size=batch_size,
                                                       directory=val_dir,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary') 

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Create the model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.1),
    Flatten(),
    Dense(512, activation='softmax'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=total_val // batch_size,
)

model.summary()

# See the results
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()