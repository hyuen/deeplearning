#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#writer = tf.train.SummaryWriter('/tmp/', graph=tf.get_default_graph())


def manual_backprop():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    a_0 = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    middle = 10
    w_1 = tf.Variable(tf.truncated_normal([784, middle]))
    b_1 = tf.Variable(tf.truncated_normal([middle]))

    def sigma(x):
        return tf.div(tf.constant(1.0),
                      tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

    def sigmaprime(x):
        return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

    z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
    a_1 = sigma(z_1)

    # to compute gradients
    diff = tf.subtract(a_1, y)

    d_z_1 = tf.multiply(diff, sigmaprime(z_1))
    d_b_1 = d_z_1
    d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

    eta = tf.constant(0.01)

    step = [
        tf.assign(w_1,
                  tf.subtract(w_1, tf.multiply(eta, d_w_1))),
        tf.assign(b_1,
                  tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(d_b_1, axis=[0]))))
    ]

    acct_mat = tf.equal(tf.argmax(a_1, 1), tf.argmax(y, 1))
    acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

    config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in xrange(10000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            _, summary = sess.run(step, feed_dict = {a_0: batch_xs,
                                        y : batch_ys})
            if i % 1000 == 0:
                print summary
                res = sess.run(acct_res, feed_dict =
                               {a_0: mnist.test.images[:1000],
                                y : mnist.test.labels[:1000]})
                print res


def sgd_backprop():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    a_0 = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    middle = 10
    w_1 = tf.Variable(tf.truncated_normal([784, middle]))
    b_1 = tf.Variable(tf.truncated_normal([middle]))

    z_1 = tf.matmul(a_0, w_1) + b_1
    a_1 = tf.nn.softmax(z_1)

    acct_mat = tf.equal(tf.argmax(a_1, 1), tf.argmax(y, 1))
    acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32)) #/tf.size(a_1)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a_1), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    config = tf.ConfigProto(device_count={'GPU': 0})

    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()

    for i in xrange(10000):
        batch_xs, batch_ys = mnist.train.next_batch(10)
        sess.run(train_step, feed_dict = {a_0: batch_xs,
                                    y : batch_ys})
        if i % 1000 == 0:
            res = sess.run(acct_res, feed_dict =
                           {a_0: mnist.test.images[:1000],
                            y : mnist.test.labels[:1000]})
            print i, res


manual_backprop()
sgd_backprop()