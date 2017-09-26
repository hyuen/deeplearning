import tensorflow as tf
from  tensorflow.contrib.lookup import MutableHashTable
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

with tf.Session() as sess:
  ht = MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=66)

  #k = tf.placeholder(tf.string, [3])
  #v = tf.placeholder(tf.int64, [3])
  key = constant_op.constant(['key'])
  value = constant_op.constant([42], dtype=dtypes.int64)
  ht.init()
  print "ht", dir(ht)
  v2 = ht.insert(key, value)


  v2 = ht.lookup(key)
  #ht.init.run()
  #init = tf.global_variables_initializer()
  #sess.run(init)
  #print v2
  print "v2", dir(v2)
  print v2.eval()

