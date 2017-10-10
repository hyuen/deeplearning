#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.lookup import MutableHashTable
import sys

def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


def dense():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    w = tf.Variable(tf.truncated_normal([784, 10]))
    b = tf.Variable(tf.truncated_normal([10]))

    z = tf.add(tf.matmul(x, w), b)

    yhat = tf.nn.softmax(z)

    L = tf.reduce_sum(-y * tf.log(yhat))

    print tf.trainable_variables()
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(L)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in xrange(10000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            # print batch_xs.shape, batch_ys.shape
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            # print tf.shape(L)
            if i % 1000 == 0:
                print i, sess.run(L, feed_dict={x: batch_xs, y: batch_ys})
            # print sess.run(L)


def dense_manual():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    w = tf.Variable(tf.truncated_normal([784, 10]))
    b = tf.Variable(tf.truncated_normal([10]))

    z = tf.add(tf.matmul(x, w), b)

    yhat = sigma(z)

    L = -tf.reduce_sum(y * tf.log(yhat))

    d_yhat = -(tf.div(y,yhat))
    d_z = d_yhat * sigmaprime(z)

    d_b = d_z
    d_w = tf.matmul(x, d_z, transpose_a=True)

    delta = tf.constant(0.0001)

    print "shapes", tf.shape(z), tf.shape(d_z), tf.shape(d_b), tf.shape(d_w)

    step = [
        tf.assign(w, tf.subtract(w, tf.multiply(delta, d_w))),
        tf.assign(b, tf.subtract(b, tf.multiply(delta, tf.reduce_sum(d_b, axis=[0])))) # why reduce mean?
        ]

    print tf.trainable_variables()

    config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        print "s, ", tf.shape(w), tf.shape(b)

        for i in xrange(10000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(step, feed_dict={x: batch_xs, y: batch_ys})
            if i % 1000 == 0:
                print i, sess.run(L, feed_dict={x: batch_xs, y: batch_ys})



def get_w_updates_forward(ht, w, x):
    # returns only the cols that need to be forward propagated
    cols = tf.constant([[1],[3],[9]], tf.int32) # computed through ht,x
    return cols

def get_b_updates_forward(ht, x):
    pass

def get_z_updates_backward(ht, x):
    # returns only the cols that need to be forward propagated
    pass

def sparse():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    w = tf.Variable(tf.truncated_normal([784, 10]), name='w')
    b = tf.Variable(tf.truncated_normal([10]), name='b')

    #z_d = tf.placeholder(tf.float32, [None, 10])
    #z_d = tf.Variable(tf.truncated_normal([1,10]), name='z_d')

    ht = MutableHashTable(key_dtype=tf.string,
                          value_dtype=tf.int64,
                          default_value=-1)

    # Forward Path
    #
    # z = w * x + b
    # yhat = sigma(z)
    # diff = yhat - y

    # init ht, should it be after one epoch?
    # cols = ht(x)
    # z_1 = sparse(w, cols) * x + sparse(b, cols)
    # z = update(z, z_1, cols)
    # yhat_1 = sigma(z_1)
    # yhat = update(yhat, yhat_1)
    # diff = yhat - y


    # Backprop
    #
    #


    # Dense section
    #forward
    z_d = tf.add(tf.matmul(x, w), b)
    #tf.assign(z_d, tf.add(tf.matmul(x, w), b))
    yhat_d = sigma(z_d)

    diff = y - yhat_d
    d_z = tf.multiply(diff, sigmaprime(z_d))

    L = -tf.reduce_sum(y * tf.log(yhat_d))

    # backward
    d_yhat_d = -(tf.div(y, yhat_d))
    d_z_d = d_yhat_d * sigmaprime(z_d)

    d_b_d = d_z_d
    d_w_d = tf.matmul(x, d_z_d, transpose_a=True)

    ####################################################################3
    # Sparse section
    # forward
    hot_cols = get_w_updates_forward(ht, w, x)
    print "---hot columns", hot_cols.get_shape()
    data = tf.transpose(tf.gather_nd(tf.transpose(w), hot_cols))
    w_1 = tf.IndexedSlices(data, hot_cols, dense_shape=tf.shape(w))
    print "---submatrix w_1", w_1.values.get_shape(), "original", w.get_shape()

    data2 = tf.transpose(tf.gather_nd(tf.transpose(b), hot_cols))
    b_1 = tf.IndexedSlices(data2, hot_cols, dense_shape=tf.shape(b))
    print "---b b_1", b_1.values.get_shape(), "original", b.get_shape()

    z_1 = tf.add(tf.matmul(x, w_1.values), b_1.values)
    print "--post_z", z_1.get_shape(), z_d.get_shape()

    print "before running", z_d, w, w_1

    z_11 = tf.scatter_nd(hot_cols, tf.transpose(z_1), tf.shape(z_d))
    print "foo---", z_11.get_shape()

    #z_11 = tf.sparse_to_dense(hot_cols, tf.shape(z_d), tf.transpose(z_1))
    #z_d = tf.scatter_update(z_d, hot_cols, tf.transpose(z_1))
    z_d = z_11


    print "---sparse2dense", z_11.get_shape(), z_1.get_shape(), z_d.get_shape()
    #z_d += z_11
    #z_d = tf.scatter_update(z_d, hot_cols, tf.transpose(z_1))

    yhat = sigma(z_d)

    diff = y - yhat
    d_z = tf.multiply(diff, sigmaprime(z_d))
    d_b = d_z
    d_w = tf.matmul(x, d_z, transpose_a=True)
    # backward


    delta = tf.constant(0.0001)

    #print "shapes", tf.shape(z), tf.shape(d_z), tf.shape(d_b), tf.shape(d_w)


    dense_step = [
        tf.assign(w, tf.subtract(w, tf.multiply(delta, d_w_d))),
        tf.assign(b, tf.subtract(b, tf.multiply(delta, tf.reduce_mean(d_b_d, axis=[0]))))
    ]

    populate_ht = [


    ]

    sparse_step = [
        # forward
        # cols = ht(x)
        # z_1 = sparse(w, cols) * x + sparse(b, cols)
        # z = update(z, z_1, cols)
        # yhat_1 = sigma(z_1)
        # yhat = update(yhat, yhat_1)
        # diff = yhat - y

        tf.assign(w, tf.subtract(w, tf.multiply(delta, d_w))),
        tf.assign(b, tf.subtract(b, tf.multiply(delta, tf.reduce_mean(d_b, axis=[0]))))
    ]

    print tf.trainable_variables()

    #sys.exit(0)
    config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('/tmp/1', graph=sess.graph)

        tf.global_variables_initializer().run()
        print "s, ", tf.shape(w), tf.shape(b)

        batch_xs, batch_ys = mnist.train.next_batch(10)
        z_1_e, z_11_e, z_d_e = sess.run([z_1, z_11, z_d], feed_dict={x: batch_xs, y: batch_ys})
        #print "ff", z_1_e, z_1_e.shape
        #print "fg", z_11_e, z_11_e.shape
        #print "fh", z_d_e, z_d_e.shape

        ndense = 100
        nsparse = 100000

        # do the first iteration here, then update the hash table
        print "running %d dense steps first" % ndense
        for i in xrange(ndense):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(dense_step, feed_dict={x: batch_xs, y: batch_ys})

        sess.run(populate_ht)
        print "-----xsparse2dense", tf.shape(z_11), tf.shape(z_d)

        # do the rest of the iterations

        print "-----dtype=running", tf.shape(w_1.values), w_1.values.dtype

        print "running %d sparse steps" % nsparse
        report_period_nsparse = nsparse / 100
        for i in xrange(nsparse):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            #print "shape", batch_xs.shape
            sess.run(sparse_step, feed_dict={x: batch_xs, y: batch_ys})
            if i % nsparse == 0 and nsparse > 0:
            #print diff.eval()
                print i, np.sum(sess.run(diff, feed_dict={x: batch_xs, y: batch_ys}))
                pass

#dense()
#dense_manual()
sparse()