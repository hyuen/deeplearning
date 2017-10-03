import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.lookup import MutableHashTable

def get_values(x, ht):

def sparse_test():
    config = tf.ConfigProto(device_count={'GPU': 0})
    size = 10

    x = tf.placeholder(tf.float32, [None, size], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")

    w = tf.Variable(tf.truncated_normal([size, size]), name="v1")
    b = tf.Variable(tf.truncated_normal([size, size]), name="v2")

    ht = MutableHashTable()

    cols, values = get_values(x, ht)

    w = tf.scatter_update(w, cols, values)



    z = tf.matmul(a_0, v1)

    delta =  tf.gather_nd(v2, [[1],[8]])
    v1 = tf.scatter_update(v1, [1,3], delta)
    v1_view = tf.gather_nd(v1, [[1], [8]])

    loss = v2 - v1

    tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('/tmp/1', graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        print "v2", tf.shape(v2).eval()
        print "delta", tf.shape(delta).eval(), delta.eval()

        print "v1_view", v1_view.eval()

        #v1 = tf.scatter_update(v1, [1,3], delta)
        #v1_view = tf.gather_nd(v1, [[1], [8]])

        print "v1_view", v1_view.eval()

sparse_test()