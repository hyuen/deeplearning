#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.lookup import MutableHashTable


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



def get_z_updates_forward(ht, x):
    # returns only the cols that need to be forward propagated
    pass

def get_z_updates_backward(ht, x):
    # returns only the cols that need to be forward propagated
    pass

def sparse():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    w = tf.Variable(tf.truncated_normal([784, 10]))
    b = tf.Variable(tf.truncated_normal([10]))


    ht = MutableHashTable(key_dtype=tf.string,
                          value_dtype=tf.int64,
                          default_value=-1)

    cols, values = get_z_updates_forward(ht, x)


    #z__1 =
    z = tf.add(tf.matmul(x, w), b)



    yhat = sigma(z)

    diff = y - yhat
    d_z = tf.multiply(diff, sigmaprime(z))

    # L = tf.reduce_sum(-y * tf.log(yhat))
    # d_L = tf.reduce_sum(-tf.div(y,yhat))
    # d_z = d_L * sigmaprime(z)

    d_b = d_z
    d_w = tf.matmul(x, d_z, transpose_a=True)

    delta = tf.constant(0.0001)

    print "shapes", tf.shape(z), tf.shape(d_z), tf.shape(d_b), tf.shape(d_w)

    step = [
        tf.assign(w, tf.subtract(w, tf.multiply(delta, d_w))),
        tf.assign(b, tf.subtract(b, tf.multiply(delta, tf.reduce_mean(d_b, axis=[0]))))  # why reduce mean?
    ]

    print tf.trainable_variables()

    config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        print "s, ", tf.shape(w), tf.shape(b)

        for i in xrange(1000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(step, feed_dict={x: batch_xs, y: batch_ys})
            # if i % 1000 == 0:
            # print i, sess.run([b, z, yhat, y, L], feed_dict={x: batch_xs, y: batch_ys})
            print i, np.sum(sess.run([diff], feed_dict={x: batch_xs, y: batch_ys}))

#dense()
dense_manual()
