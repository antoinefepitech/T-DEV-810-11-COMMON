# Useful librairies
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, time, datetime, shutil, os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Our model
from models.VggModel import VggModel, get_model


# Global variables
BATCH_SIZE = 50
NB_EPOCHS = 100
IMG_WIDTH = 96
IMG_HEIGHT = 96
TRAIN_DATA_PATH = 'chest_xray/train'
TEST_DATA_PATH = 'chest_xray/test'
VAL_DATA_PATH = 'chest_xray/val'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
AUTOTUNE = tf.data.experimental.AUTOTUNE
VERBOSE = 1

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

def save_model(model=None, model_name='vgg16'):
  """
  Save a TF Model into h5 format
  """
  model.save('saved_model/{}/model.h5'.format(model_name))
  print("Model saved successfully.")

def get_callbacks(model_size='small'):
  """
  Define the callbacks for the ML model
  """
  return [
    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
    tf.keras.callbacks.TensorBoard(os.path.join("logs/{}".format('vgg16'), datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)
  ]

def get_label(file_path):
  """
  Get the label of a file - Can be NORMAL or PNEUMONIA
  """
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  """
  Convert an image into a tensor with needed size
  """
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  """
  Process a file
  """
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
  
def main():
  """
  Main function
  """
  # To get the nb of steps and how many images we got
  nb_normal_tr = len(os.listdir('{}/NORMAL'.format(TRAIN_DATA_PATH)))
  nb_pneumonia_tr = len(os.listdir('{}/PNEUMONIA'.format(TRAIN_DATA_PATH)))
  nb_normal_val = len(os.listdir('{}/NORMAL'.format(VAL_DATA_PATH)))
  nb_pneumonia_val = len(os.listdir('{}/PNEUMONIA'.format(VAL_DATA_PATH)))
  total_train = nb_normal_tr + nb_pneumonia_tr
  total_val = nb_normal_val + nb_pneumonia_val

  # Our datas generators
  train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
  validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
  test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

  train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=TRAIN_DATA_PATH,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
  )
  val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=VAL_DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
  )
  test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=TEST_DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
  )

  # Get the model
  model = get_model(
    model='vgg',
    nodes=16,
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    hidden_activation='relu',
    final_activation='softmax',
    metrics=None
  )

  # Train the model
  model.fit(
    train_data_gen,
    callbacks=get_callbacks(),
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=NB_EPOCHS,
    validation_data=val_data_gen,
    validation_steps=total_val // BATCH_SIZE
  )

  model.summary()

  save_model(model, 'vgg16')

  # Use a testing model to display metrics
  testing_model = keras.Sequential([model, keras.layers.Softmax()])
  testing_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=METRICS
  )

  # Display metrics for testing purpose
  print('Normal/Pneumonia vgg trained model : ')
  results = testing_model.evaluate(test_data_gen)
  for name, value in zip(model.metrics_names, results):
    print(f'{name} : {value}')

if __name__ == "__main__":
  main()
  exit()