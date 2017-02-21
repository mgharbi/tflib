import tensorflow as tf
import numpy as np


def l2_loss(target, prediction, name=None):
  with tf.name_scope(name, default_name='l2_loss', values=[target, prediction]):
    loss = tf.reduce_mean(tf.square(target-prediction))
  return loss


def l1_loss(target, prediction, name=None):
  with tf.name_scope(name, default_name='l1_loss', values=[target, prediction]):
    loss = tf.reduce_mean(tf.abs(target-prediction))
  return loss


def binary_classification_loss(predictions, labels, name=None):
  with tf.name_scope(name, default_name='binary_class_loss', values=[predictions, labels]):
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=labels))
  return loss


def psnr(target, prediction, name=None):
  with tf.name_scope(name, default_name='psnr_op', values=[target, prediction]):
    squares = tf.square(target-prediction, name='squares')
    squares = tf.reshape(squares, [tf.shape(squares)[0], -1])
    # mean psnr over a batch
    p = tf.reduce_mean((-10/np.log(10))*tf.log(tf.reduce_mean(squares, axis=[1])))
  return p


def global_affine_invariant_mse(im1, im2, name=None):
  """Global affine invariant MSE.

  im2 is the reference image.
  computes \min_{A} ||im1*A - im2||^2
  """

  with tf.name_scope(name, default_name='ai_mse', values=[im1, im2]):
    bs, _, _, c = im1.get_shape().as_list()
    ones = tf.expand_dims(tf.ones_like(im1[:, :, :, 0]), 3)
    im1 = tf.concat((im1, ones), 3)

    im1 = tf.reshape(im1, [bs, -1, c+1])
    im2 = tf.reshape(im2, [bs, -1, c])

    affine_mtx = tf.matrix_solve_ls(im1, im2)

    err = tf.square(tf.matmul(im1, affine_mtx)-im2)
    err = tf.reduce_mean(err)
    return err


def global_affine_invariant_psnr(im1, im2, peak_val=1.0, name=None):
  with tf.name_scope(name, default_name='ai_psnr', values=[im1, im2]):
    err = global_affine_invariant_mse(im1, im2)
    p = -10*tf.log(err/(peak_val*peak_val))/tf.log(10.0)
  return p
