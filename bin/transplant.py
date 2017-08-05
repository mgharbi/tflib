#!/usr/bin/env python
# encoding: utf-8
# -------------------------------------------------------------------
# File:    train.py
# Author:  Michael Gharbi <gharbi@mit.edu>
# Created: 2016-10-25
# -------------------------------------------------------------------
# 
# 
# 
# ------------------------------------------------------------------#
"""Train a model."""

import argparse
import os
import time

import numpy as np

import caffe
import tensorflow as tf


def main(args):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    src_path = tf.train.latest_checkpoint(args.src)
    if src_path is None:
      print ('Transplant: could not find a checkpoint in {}'.format(args.src))
      return

    # Load src_path and metagraph, list all vars
    metapath = ".".join([src_path, "meta"])
    saver = tf.train.import_meta_graph(metapath)
    saver.restore(sess, src_path)

    src_vars = set([v.name for v in tf.global_variables()])

  tf.reset_default_graph()
  with tf.Session(config=config) as sess:
    # Load dst_path and metagraph, list all vars
    dst_path = tf.train.latest_checkpoint(args.dst)
    if dst_path is None:
      print ('Transplant: could not find a checkpoint in {}'.format(args.dst))
      return
    metapath = ".".join([dst_path, "meta"])
    saver = tf.train.import_meta_graph(metapath)
    saver.restore(sess, dst_path)

    dst_vars = set([v.name for v in tf.global_variables()])

    # intersect vars
    vars_to_load = dst_vars.intersection(src_vars)

    # ignore some more vars
    if args.ignore_variables is not None:
      ignores = set([v.strip() for v in args.ignore_variables.split(',')])
      vars_to_load = vars_to_load.difference(ignores)

    vars_to_load = [tf.get_default_graph().get_tensor_by_name(v) for v in vars_to_load]

    # load variables
    src_saver = tf.train.Saver(vars_to_load)
    dst_saver = tf.train.Saver(tf.global_variables())

    print "loading src checkpoint {} with {} variables".format(src_path, len(vars_to_load))
    src_saver.restore(sess, src_path)

    # write new chkpt
    new_path = os.path.join(os.path.dirname(dst_path), 'model.ckpt')
    print "saving checkpoint:", new_path
    if not os.path.exists(args.dst):
      os.makedirs(args.dst)
    dst_saver.save(sess, new_path, global_step=0)

    print "Vars loaded:"
    for v in vars_to_load:
      print v.name

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src', type=str)
  parser.add_argument('--dst', type=str)
  parser.add_argument('--ignore_variables', default=None, type=str, help= 'comma separated list of variables to ignore')
  args = parser.parse_args()
  main(args)
