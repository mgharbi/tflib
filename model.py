import abc
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

class BaseModel(object):
  """Base model implementing a training loop and general model interface."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, inputs, is_training=False, reuse=False):
    """Creates a model mapped to a directory on disk for I/O:

    Args:
      inputs: input tensor(s), can be placeholders (e.g. for runtime prediction) or 
              a queued data_pipeline.
      is_training: allows to parametrize certain layers differently when training (e.g. batchnorm).
      reuse: whether to reuse weights defined by another model.
    """

    self.inputs = inputs
    self.is_training = is_training
    self.reuse = reuse

    self.layers = {}
    self.summaries = []

    self.global_step = tf.Variable(
        0, name='global_step', trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    self._setup_prediction()

  @abc.abstractmethod
  def _setup_prediction(self):
    """Core layers for model prediction."""
    pass

  @abc.abstractmethod
  def _setup_loss(self):
    """Loss function to minimize."""
    pass

  @abc.abstractmethod
  def _setup_optimizer(self, learning_rate):
    """Optimizer."""
    pass

  @abc.abstractmethod
  def _tofetch(self):
    """Tensors to run/fetch at each training step.
    Returns:
      tofetch: (dict) of Tensors/Ops.
    """
    pass

  def _train_step(self, sess, run_options=None, run_metadata=None):
    """Step of the training loop.

    Returns:
      data (dict): data from useful for printing in 'summary_step'.
                   Should contain field "step" with the current_step.
    """
    tofetch = self._tofetch()
    tofetch['step'] = self.global_step
    tofetch['summaries'] = self.merged_summaries
    start_time = time.time()
    data = sess.run(tofetch, options=run_options, run_metadata=run_metadata)
    data['duration'] = time.time()-start_time
    return data

  @abc.abstractmethod
  def _summary_step(self, data):
    """Information form data printed at each 'summary_step'.

    Returns:
      message (str): string printed at each summary step.
    """
    pass

  def load(self, sess, saver, checkpoint_dir):
    """Loads the latest checkpoint from disk.

    Args:
      sess (tf.Session): current session in which the parameters are imported.
    """
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
      print 'Model not loaded, checkpoint not found: {}'.format(checkpoint_path)
      return False
    else:
      saver.restore(sess, checkpoint_path)
      step = tf.train.global_step(sess, self.global_step)
      print 'Loaded model at step {} from snapshot {}.'.format(step, checkpoint_path)
      return True

  def save(self, sess, saver, checkpoint_dir):
    """Saves a checkpoint to disk.

    Args:
      sess (tf.Session): current session from which the parameters are saved.
    """
    checkpoint_path = os.path.join(checkpoint_dir, 'model')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver.save(sess, checkpoint_path, global_step=self.global_step)

  def train(self, learning_rate, checkpoint_dir,
      sess=None, saver=None, 
      summary_step=100, checkpoint_step=1000, profiling=False):
    """Main training loop.

    Args:
      learning_rate (float): global learning rate used for the optimizer.
      summary_step (int): frequency at which log entries are added.
      checkpoint_step (int): frequency at which checkpoints are saved to disk.
      profiling: whether to save profiling trace at each summary step. (used for perf. debugging).
    """
    lr = tf.Variable(learning_rate, name='learning_rate',
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    self.summaries.append(tf.summary.scalar('learning_rate', lr))

    if saver is None:
      saver = tf.train.Saver(tf.global_variables())

    # Optimizer
    self._setup_loss()
    self._setup_optimizer(lr)

    # Profiling
    if profiling:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
    else:
      run_options = None
      run_metadata = None

    # Summaries
    self.merged_summaries = tf.summary.merge(self.summaries)

    if sess is None:
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)

    with sess.as_default():
      self.summary_writer = tf.summary.FileWriter(checkpoint_dir, graph=sess.graph)

      print 'Initializing all variables.'
      tf.local_variables_initializer().run()
      tf.global_variables_initializer().run()

      # Load or start from scratch?
      has_chkpt = self.load(sess, saver, checkpoint_dir)
      if not has_chkpt:
        print 'No checkpoint found, training from scratch.'

      print 'Starting data threads coordinator.'
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      print 'Starting optimization.'
      try:
        while not coord.should_stop():  # Training loop
          step_data = self._train_step(sess, run_options, run_metadata)
          step = step_data['step']

          if step > 0 and step % summary_step == 0:
            if profiling:
              self.summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
              tl = timeline.Timeline(run_metadata.step_stats)
              ctf = tl.generate_chrome_trace_format()
              with open(os.path.join(checkpoint_dir, 'timeline.json'), 'w') as fid:
                print ('Writing trace.')
                fid.write(ctf)

            print self._summary_step(step_data)
            self.summary_writer.add_summary(step_data['summaries'], global_step=step)

          # Save checkpoint every `checkpoint_step`
          if checkpoint_step is not None and (
              step > 0) and step % checkpoint_step == 0:
            print 'Step {} | Saving checkpoint.'.format(step)
            self.save(sess, saver, checkpoint_dir)

      except KeyboardInterrupt:
        print 'Interrupted training at step {}.'.format(step)
        self.save(sess, saver, checkpoint_dir)

      except tf.errors.OutOfRangeError:
        print 'Training completed at step {}.'.format(step)
        self.save(sess, saver, checkpoint_dir)

      finally:
        print 'Shutting down data threads.'
        coord.request_stop()
        self.summary_writer.close()

      # Wait for data threads
      print 'Waiting for all threads.'
      coord.join(threads)

      print 'Optimization done.'
    sess.close()
