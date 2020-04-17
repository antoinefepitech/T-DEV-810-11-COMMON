import tensorflow as tf

from tensorflow.keras import layers


class BasicBlock(layers.Layer):
  """
  BasicBlock class
  """
  def __init__(self, filter_num, stride=1):
    super(BasicBlock, self).__init__()
    self.conv_b_1 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
    self.bn_b_1 = layers.BatchNormalization()
    self.conv_b_2 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')
    self.bn_b_2 = layers.BatchNormalization()
    if stride != 1:
      self.downsample = tf.keras.Sequential()
      self.downsample.add(layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride))
      self.downsample.add(layers.BatchNormalization())
    else:
      self.downsample = lambda x: x

  def call(self, inputs, training=None, **kwargs):
    residual = self.downsample(inputs)
    x = self.conv_b_1(inputs)
    x = self.bn_b_1(x, training=training)
    x = tf.nn.relu(x)
    x = self.conv_b_2(x)
    x = self.bn_b_2(x, training=training)
    output = tf.nn.relu(layers.add([residual, x]))
    return output


class BottleNeck(layers.Layer):
  """
  Bottleneck class
  """
  def __init__(self, filter_num, stride=1):
    super(BottleNeck, self).__init__()
    self.conv_b_1 = layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same')
    self.bn_b_1 = layers.BatchNormalization()
    self.conv_b_2 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
    self.bn_b_2 = layers.BatchNormalization()
    self.conv_b_3 = layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same')
    self.bn_b_3 = layers.BatchNormalization()

    self.downsample = tf.keras.Sequential()
    self.downsample.add(layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride))
    self.downsample.add(layers.BatchNormalization())

  def call(self, inputs, training=None, **kwargs):
    residual = self.downsample(inputs)
    x = self.conv_b_1(inputs)
    x = self.bn_b_1(x, training=training)
    x = tf.nn.relu(x)
    x = self.conv_b_2(x)
    x = self.bn_b_2(x, training=training)
    x = tf.nn.relu(x)
    x = self.conv_b_3(x)
    x = self.bn_b_3(x, training=training)
    output = tf.nn.relu(layers.add([residual, x]))
    return output


def make_basic_block_layer(filter_num, blocks, stride=1):
  """
  Generate a block layer
  """
  res_block = tf.keras.Sequential()
  res_block.add(BasicBlock(filter_num, stride=stride))

  for _ in range(1, blocks):
    res_block.add(BasicBlock(filter_num, stride=1))

  return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
  """
  Generate a bottleneck layer
  """
  res_block = tf.keras.Sequential()
  res_block.add(BottleNeck(filter_num, stride=stride))

  for _ in range(1, blocks):
    res_block.add(BottleNeck(filter_num, stride=1))

  return res_block