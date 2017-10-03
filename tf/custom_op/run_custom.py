import tensorflow as tf


custom_op = tf.load_op_library('./custom_op.so')

with tf.Session(''):
    custom_op.zero_out([[1,2,3]]).eval()