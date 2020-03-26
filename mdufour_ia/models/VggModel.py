from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

IMG_WIDTH = 96
IMG_HEIGHT = 96

class VggModel(tf.keras.Model):
  """
  A VggModel - CF https://arxiv.org/pdf/1409.1556/
  """
  def __init__(self, name=None, nodes=1, hidden_activation='linear', final_activation='softmax'):
    super(VggModel, self).__init__(name=name)
    # first conv
    self.conv_1 = layers.Conv2D(16, 3, padding='same', activation=hidden_activation, input_shape=(IMG_HEIGHT, IMG_WIDTH , 3))
    self.pooling_1 = layers.MaxPooling2D()
    # second conv
    self.conv_2 = layers.Conv2D(32, 3, padding='same', activation=hidden_activation)
    self.pooling_2 = layers.MaxPooling2D()
    # thrid conv
    self.conv_3 = layers.Conv2D(64, 3, padding='same', activation=hidden_activation)
    self.pooling_3 = layers.MaxPooling2D()
    # flatten layer
    self.flatten_1 = layers.Flatten()
    # dense layer
    self.dense_1 = layers.Dense(512, activation=hidden_activation)
    # pref layer
    self.pred_layer = layers.Dense(1, activation=final_activation, name='predictions')

  def call(self, inputs):
    x = self.conv_1(inputs)
    x = self.pooling_1(x)
    x = self.conv_2(x)
    x = self.pooling_2(x)
    x = self.conv_3(x)
    x = self.pooling_3(x)
    x = self.flatten_1(x)
    x = self.dense_1(x)
    return self.pred_layer(x)


def get_model(model='vgg16', nodes=16, optimizer='adam', loss='binary_crossentropy', hidden_activation='linear', final_activation='softmax', metrics='accuracy'):
  """
  Return a basic model
  """

  model = VggModel(name='vgg_model', nodes=nodes, hidden_activation=hidden_activation, final_activation=final_activation)

  # model = tf.keras.Sequential([
  #   layers.Conv2D(16, 3, padding='same', activation=hidden_activation, input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
  #   layers.MaxPooling2D(),
  #   # layers.Dropout(0.25),
  #   layers.Conv2D(32, 3, padding='same', activation=hidden_activation),
  #   layers.MaxPooling2D(),
  #   # layers.Dropout(0.25),
  #   layers.Conv2D(64, 3, padding='same', activation=hidden_activation),
  #   layers.MaxPooling2D(),
  #   # layers.Dropout(0.25),
  #   layers.Flatten(),
  #   layers.Dense(512, activation=final_activation),
  #   layers.Dense(3)
  # ])

  model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
  return model

