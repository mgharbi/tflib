import os

import numpy as np
import tensorflow as tf
import tempfile

import trainer

class TrainerTest(tf.test.TestCase):

  def test_optimize(self):
    v = tf.Variable((3,), dtype=tf.float32, name='var')
    target = tf.ones_like(v)
    with tf.name_scope('loss'):
      loss = tf.reduce_sum(tf.square(target-v))
    lr = 1e-1

    global_step = tf.Variable(
        0, name='global_step', trainable=False,
        collections=['global_step', tf.GraphKeys.GLOBAL_VARIABLES])

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(
        loss, global_step=global_step)

    fetches = {"loss":loss}

    t = trainer.Trainer(optimizer, global_step, fetches=fetches)

    checkpoint_dir = tempfile.mkdtemp()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      t.train(sess, checkpoint_dir, max_step=100, summary_step=10)
      v_ = sess.run(v)
      assert np.sum(np.square(v_-1)) < 1e-4
