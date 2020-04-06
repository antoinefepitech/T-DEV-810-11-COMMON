from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from models.residual_block import make_basic_block_layer, make_bottleneck_layer

IMG_WIDTH = 96
IMG_HEIGHT = 96
NUM_CLASSES = 3

class ResNetTypeI(tf.keras.Model):
  """
  A ResNetTypeI Model - Use of basic bloc layers (conv2D + batch normalization + conv2D + batch normalization)
  """
  def __init__(self, nodes, name=None, final_activation='softmax'):
    super(ResNetTypeI, self).__init__(name=name)
    # first conv
    self.conv_1 = layers.Conv2D(16, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3), strides=2)
    # batch normalization
    self.bn_1 = layers.BatchNormalization()
    # pooling
    self.pooling_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
    # basic block layers
    self.layer_1 = make_basic_block_layer(filter_num=64, blocks=nodes[0])
    self.layer_2 = make_basic_block_layer(filter_num=128, blocks=nodes[1], stride=2)
    self.layer_3 = make_basic_block_layer(filter_num=256, blocks=nodes[2], stride=2)
    self.layer_4 = make_basic_block_layer(filter_num=512, blocks=nodes[3], stride=2)
    # global average pool
    self.avg_pool = layers.GlobalAveragePooling2D()
    # prediction
    self.pred_layer = layers.Dense(units=NUM_CLASSES, activation=final_activation, name='predictions')

  def call(self, inputs):
    x = self.conv_1(inputs)
    x = self.bn_1(x)
    x = tf.nn.relu(x)
    x = self.pooling_1(x)
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    x = self.avg_pool(x)
    return self.pred_layer(x)

class ResNetTypeII(tf.keras.Model):
  """
  A ResNetTypeII Model - Use of bottleneck layers (conv2D + batch normalization + conv2D + batch normalization)
  """
  def __init__(self, nodes, name=None, final_activation='softmax'):
    super(ResNetTypeII, self).__init__(name=name)
    # first conv
    self.conv_1 = layers.Conv2D(16, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3), strides=2)
    # batch normalization
    self.bn_1 = layers.BatchNormalization()
    # pooling
    self.pooling_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
    # basic block layers
    self.layer_1 = make_bottleneck_layer(filter_num=64, blocks=nodes[0])
    self.layer_2 = make_bottleneck_layer(filter_num=128, blocks=nodes[1], stride=2)
    self.layer_3 = make_bottleneck_layer(filter_num=256, blocks=nodes[2], stride=2)
    self.layer_4 = make_bottleneck_layer(filter_num=512, blocks=nodes[3], stride=2)
    # global average pool
    self.avg_pool = layers.GlobalAveragePooling2D()
    # prediction
    self.pred_layer = layers.Dense(units=NUM_CLASSES, activation=final_activation, name='predictions')

  def call(self, inputs):
    x = self.conv_1(inputs)
    x = self.bn_1(x)
    x = tf.nn.relu(x)
    x = self.pooling_1(x)
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    x = self.avg_pool(x)
    return self.pred_layer(x)

def get_model_resnet(model='resnet_18', optimizer='adam', loss='binary_crossentropy', final_activation='softmax', metrics='accuracy'):
  """
  Return a basic model
  """
  if model == 'resnet_18':
    model = resnet_18(final_activation=final_activation)
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model
  elif model == 'resnet_34':
    model = resnet_34(final_activation=final_activation)
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model

def resnet_18(final_activation):
    return ResNetTypeI(nodes=[2, 2, 2, 2], name='resnet1_18', final_activation=final_activation)


def resnet_34(final_activation):
    return ResNetTypeI(nodes=[3, 4, 6, 3], name='resnet1_34', final_activation=final_activation)


