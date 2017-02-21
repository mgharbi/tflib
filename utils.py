import tensorflow as tf

def get_model_params(sess, param_collection="model_params"):
  pcoll = tf.get_collection(param_collection)
  params_ = {p.name.split(':')[0]: p for p in pcoll}
  model_params = sess.run(params_)
  return model_params


def get_params_summaries():
  weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
  biases = tf.get_collection(tf.GraphKeys.BIASES)
  summaries = []

  for w in weights:
    name = w.name.split('/weights')[0]
    summaries.append(tf.summary.histogram('weights/{}'.format(name), w))
    if len(w.get_shape().as_list()) == 4: # 2D conv
      fh, fw, nin, nout = w.get_shape().as_list()
      wviz = tf.transpose(w, perm=[3, 0, 2, 1])
      wviz = tf.reshape(wviz, [1, nout*fh, fw*nin, 1])
      summaries.append(tf.summary.image('weights/{}'.format(name), wviz))
    elif len(w.get_shape().as_list()) == 2: # FC
      nin, nout = w.get_shape().as_list()
      wviz = tf.transpose(w, perm=[1, 0])
      wviz = tf.reshape(wviz, [1, nout, nin, 1])
      summaries.append(tf.summary.image('weights/{}'.format(name), wviz))
  for b in biases:
    name = b.name.split('/biases')[0]
    summaries.append(tf.summary.histogram('biases/{}'.format(name), b))
  return summaries
