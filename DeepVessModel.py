import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')


def max_pool(x, shape):
    return tf.nn.max_pool3d(x, ksize=shape,
                            strides=[1, 2, 2, 2, 1], padding='SAME')
def define_deepvess_architecture(x):
    # Define the DeepVess Architecture

    W_conv1a = weight_variable([3, 3, 3, 1, 32])
    b_conv1a = bias_variable([32])
    h_conv1a = tf.nn.relu(conv3d(x, W_conv1a) + b_conv1a)
    W_conv1b = weight_variable([3, 3, 3, 32, 32])
    b_conv1b = bias_variable([32])
    h_conv1b = tf.nn.relu(conv3d(h_conv1a, W_conv1b) + b_conv1b)
    W_conv1c = weight_variable([3, 3, 3, 32, 32])
    b_conv1c = bias_variable([32])
    h_conv1c = tf.nn.relu(conv3d(h_conv1b, W_conv1c) + b_conv1c)
    h_pool1 = max_pool(h_conv1c, [1, 1, 2, 2, 1])

    W_conv2a = weight_variable([1, 3, 3, 32, 64])
    b_conv2a = bias_variable([64])
    h_conv2a = tf.nn.relu(conv3d(h_pool1, W_conv2a) + b_conv2a)
    W_conv2b = weight_variable([1, 3, 3, 64, 64])
    b_conv2b = bias_variable([64])
    h_conv2b = tf.nn.relu(conv3d(h_conv2a, W_conv2b) + b_conv2b)
    h_pool2 = max_pool(h_conv2b, [1, 1, 2, 2, 1])

    W_fc1 = weight_variable([1 * 5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 5 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1 * 5 * 5 * 2])
    b_fc2 = bias_variable([1 * 5 * 5 * 2])
    h_fc1 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return tf.reshape(h_fc1, [-1, 1 * 5 * 5, 2]), keep_prob