import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

class Trainer(object):
  """Implements a training loop"""

  def __init__(self, train_op, global_step, summaries=None,
      fetches=None, message_func=None):
    """
    Args:
      train_op (tf.op):
      global_step (tf.Variable):
      summaries (tf.summaries.merge([])):
      fetches (dict):
      message_func (func: dict -> str):
    """

    self.summaries = summaries

    if fetches is None:
      fetches = {}
    else:
      self.fetches = fetches
    self.fetches['global_step'] = global_step
    self.fetches['train_op'] = train_op
    
    if self.summaries is not None:
      self.fetches['summaries'] = tf.summary.merge(summaries)

    if message_func is None:
      self.message_func = lambda data: "Step {}".format(data['global_step'])
    else:
      self.message_func = message_func


  def __load(self, sess, saver, checkpoint_dir):
    """Loads the latest checkpoint from disk.

    Args:
      sess (tf.Session): current session in which the parameters are imported.
    """
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
      print 'Trainer: checkpoint not found: {}, start from scratch'.format(checkpoint_dir)
    else:
      saver.restore(sess, checkpoint_path)
      global_step = tf.get_collection('global_step')[0]
      step = tf.train.global_step(sess, global_step)
      print 'Trainer: resuming at step {} from snapshot {}.'.format(step, checkpoint_path)

  def __save(self, sess, saver, checkpoint_dir):
    """Saves a checkpoint to disk.

    Args:
      sess (tf.Session): current session from which the parameters are saved.
    """
    checkpoint_path = os.path.join(checkpoint_dir, 'model')
    if not os.path.exists(checkpoint_dir):
      print 'Train: checkpoint_dir does not exist, creating {}.'.format(checkpoint_dir)
      os.makedirs(checkpoint_dir)
    global_step = tf.get_collection('global_step')[0]
    saver.save(sess, checkpoint_path, global_step=global_step)

  def __train_step(self, sess, run_options=None, run_metadata=None):
    """Step of the training loop.

    Returns:
      data (dict): data from useful for printing in 'summary_step'.
                   Should contain field "step" with the current_step.
    """
    start_time = time.time()
    data = sess.run(self.fetches, options=run_options, run_metadata=run_metadata)
    data['duration'] = time.time()-start_time
    return data

  def train(self, sess, checkpoint_dir,
      saver=None, 
      summary_step=100, checkpoint_step=1000, max_step=None,
      profiling=False):
    """Main training loop.

    Args:
      checkpoint_dir (str): 
      sess (tf.Session): current session from which the parameters are saved.
      saver (tf.train.Saver) 
      summary_step (int): frequency at which log entries are added.
      checkpoint_step (int): frequency at which checkpoints are saved to disk.
      profiling: whether to save profiling trace at each summary step. (used for perf. debugging).
    """

    # If no specific saver, save all variables
    if saver is None:
      saver = tf.train.Saver(tf.global_variables())
     
    with sess.as_default():
      # Profiling
      if profiling:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
      else:
        run_options = None
        run_metadata = None

      summary_writer = tf.summary.FileWriter(
          checkpoint_dir, graph=sess.graph)

      print 'Trainer: initializing all variables.'
      tf.local_variables_initializer().run()
      tf.global_variables_initializer().run()

      # Load or start from scratch
      self.__load(sess, saver, checkpoint_dir)

      print 'Trainer: starting data threads coordinator.'
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      print 'Trainer: starting optimization.'
      try:
        while not coord.should_stop():  # Training loop
          step_data = self.__train_step(sess, run_options, run_metadata)
          step = step_data['global_step']

          if step > 0 and step % summary_step == 0:
            if profiling:
              summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
              tl = timeline.Timeline(run_metadata.step_stats)
              ctf = tl.generate_chrome_trace_format()
              with open(os.path.join(checkpoint_dir, 'timeline.json'), 'w') as fid:
                print ('Trainer: writing trace.')
                fid.write(ctf)

            print self.message_func(step_data)
            if self.summaries is not None:
              summary_writer.add_summary(step_data['summaries'], global_step=step)

          # Save checkpoint every `checkpoint_step`
          if checkpoint_step is not None and (
              step > 0) and step % checkpoint_step == 0:
            print 'Trainer: saving checkpoint.'.format(step)
            self.__save(sess, saver, checkpoint_dir)

          if max_step is not None and step >= max_step:
            print 'Trainer: last step, saving checkpoint.'.format(step)
            self.__save(sess, saver, checkpoint_dir)
            break

      except KeyboardInterrupt:
        print 'Trainer: interrupting training at step {}.'.format(step)
        self.__save(sess, saver, checkpoint_dir)

      except tf.errors.OutOfRangeError:
        print 'Trainer: training completed at step {}.'.format(step)
        self.__save(sess, saver, checkpoint_dir)

      finally:
        print 'Trainer: shutting down data threads.'
        coord.request_stop()
        summary_writer.close()

      # Wait for data threads
      print 'Trainer: waiting for all threads.'
      coord.join(threads)

      print 'Trainer: optimization done.'
