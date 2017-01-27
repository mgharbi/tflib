import tensorflow as tf
import numpy as np

def conv(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, default_name='conv', reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, ksize, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
      regularizer=regularizer)

    strides = [1, stride, stride, 1]
    current_layer = tf.nn.conv2d(inputs, weights, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def separable_conv(inputs,
         nfilters,
         ksize,
         stride=1,
         chan_multiplier=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, default_name='conv', reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights_d = tf.get_variable(
      'weights_depthwise',
      shape=[ksize, ksize, n_in, chan_multiplier],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
      regularizer=regularizer)
    weights_p = tf.get_variable(
      'weights_pointwise',
      shape=[1, 1, n_in*chan_multiplier, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
      regularizer=regularizer)

    strides = [1, stride, stride, 1]
    current_layer = tf.nn.separable_conv2d(inputs, weights_d, weights_p, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def depthwise_conv(inputs,
         nfilters,
         ksize,
         stride=1,
         chan_multiplier=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, default_name='conv', reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights_d = tf.get_variable(
      'weights',
      shape=[ksize, ksize, n_in, chan_multiplier],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
      regularizer=regularizer)

    strides = [1, stride, stride, 1]
    current_layer = tf.nn.depthwise_conv2d(inputs, weights_d, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def transpose_conv(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, default_name='transpose_conv', reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, ksize, nfilters, n_in],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
      regularizer=regularizer)

    bs, h, w, c = inputs.get_shape().as_list()
    strides = [1, stride, stride, 1]
    out_shape = [bs, stride*h, stride*w, nfilters] 
    current_layer = tf.nn.conv2d_transpose(inputs, weights, out_shape, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def conv3(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, default_name='conv', reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, ksize, ksize, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
      regularizer=regularizer)

    strides = [1, stride, stride, stride, 1]
    current_layer = tf.nn.conv3d(inputs, weights, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer


def fc(inputs, nfilters, use_bias=True, activation_fn=tf.nn.relu,
       initializer=tf.contrib.layers.variance_scaling_initializer(),
       regularizer=None, scope=None, reuse=None):
  with tf.variable_scope(scope, default_name='fc', reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      regularizer=regularizer)

    current_layer = tf.matmul(inputs, weights)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0))
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

  return current_layer

def batch_norm(inputs, center=False, scale=False,
               decay=0.999, epsilon=0.001, reuse=None,
               scope=None, is_training=False):
  return tf.contrib.layers.batch_norm(
    inputs, center=center, scale=scale,
    decay=decay, epsilon=epsilon, activation_fn=None,
    reuse=reuse,trainable=False, fused=False, scope=scope, is_training=is_training)

relu = tf.nn.relu

def crop_like(inputs, like, name=None):
  with tf.name_scope(name):
    _, h, w, _ = inputs.get_shape().as_list()
    _, new_h, new_w, _ = like.get_shape().as_list()
    crop_h = (h-new_h)/2
    crop_w = (w-new_w)/2
    cropped = inputs[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
    return cropped

def pixel_shuffle_upsample(im, factor, name='pixel_shuffle_upsample'):
  with tf.name_scope(name):
    bs, h, w, nfilters = im.get_shape().as_list()
    if nfilters % (factor*factor) != 0:
      raise ValueError("pixel shuffle upsample with non-integral factor!")

    nfilters /= factor*factor
    chans = tf.split(im, nfilters, axis=3, name='channels')
    new_chans = []
    for c in chans:
      c = tf.reshape(c, (bs, h, w, factor, factor), name='split')  # bs, h, w, fh, fw
      c = tf.transpose(c, (0, 1, 3, 2, 4))  # bs, h, fh, w, fw
      c = tf.reshape(c, (bs, h*factor, w*factor, 1), name='upsample')  # bs, h, fh, w, fw
      new_chans.append(c)
    ret = tf.concat(new_chans, 3)

    return ret
