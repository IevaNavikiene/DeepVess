# Copyright 2017-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu).
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#   References:
#   -----------
#   [1] Haft-Javaherian, M; Fang, L.; Muse, V.; Schaffer, C.B.; Nishimura,
#       N.; & Sabuncu, M. R. (2018) Deep convolutional neural networks for
#       segmenting 3D in vivo multiphoton images of vasculature in
#       Alzheimer disease mouse models. *arXiv preprint, arXiv*:1801.00880.
# =============================================================================

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
import h5py
import scipy.io as io
import sys,os
import itertools as it
import time
#from api.resources.preprocessing.DeepVess.TrainDeepVess import train_deep_vess
#from api.resources.preprocessing.DeepVess.DeepVessModel import define_deepvess_architecture
from tensorflow.python.client import device_lib
'''
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

available_gpus = get_available_gpus()
print(available_gpus,'available_gpus')
if len(available_gpus) != 0:
    print('using GPU', available_gpus[0].split('/device:GPU:')[1])
    available_device=available_gpus[0].split('/device:GPU:')[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = available_device
'''
import tensorflow as tf

def start_tracing_model(inputData, isTrain=False, isForward=True, padSize=((3, 3), (16, 16), (16, 16), (0, 0)), GPU_device):
    """

    :param inputData:
    :param isTrain: Change isTrain to True if you want to train the network
    :param isForward:  Change isForward to True if you want to test the network
    :param padSize: padSize is the padding around the central voxel to generate the field of view
    :return:
    """
    if GPU_device == "":
        GPU_device=1
    with tf.device('/device:GPU:' + GPU_device)
    WindowSize = np.sum(padSize, axis=1) + 1
    # pad Size aroung the central voxel to generate 2D region of interest
    corePadSize = 2
    batch_size = 1000
    # start the TF session
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)
    # create placeholder for input and output nodes
    x = tf.placeholder(tf.float32, shape=[None, WindowSize[0], WindowSize[1],
                                          WindowSize[2], WindowSize[3]])
    y_ = tf.placeholder(tf.float32, shape=[None, (2 * corePadSize + 1) ** 2, 2])

    # Import Data
    f = h5py.File(inputData, 'r')
    im = np.array(f.get('/im'))
    im = im.reshape(im.shape + (1,))
    imSize = im.size
    imShape = im.shape
    if isTrain:
        l = np.array(f.get('/l'))
        l = l.reshape(l.shape + (1,))
        nc = im.shape[1]
        tst = im[:, (nc / 2):(3 * nc / 4), :]
        tstL = l[:, (nc / 2):(3 * nc / 4), :]
        trn = im[:, 0:(nc / 2), :]
        trnL = l[:, 0:(nc / 2), :]
        tst = np.pad(tst, padSize, 'symmetric')
        trn = np.pad(trn, padSize, 'symmetric')
    if isForward:
        im = np.pad(im, padSize, 'symmetric')
        V = np.ndarray(shape=(imShape), dtype=np.float32)
    print("Data loaded.")

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def conv3d(x, W):
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')

    def max_pool(x, shape):
        return tf.nn.max_pool3d(x, ksize=shape,
                                strides=[1, 2, 2, 2, 1], padding='SAME')

    def get_batch(im, l, corePadSize, ID):
        """ generate a batch from im and l for training
            based on the location of ID entries and core pad size. Note that the ID
            is based on no core pad.
        """
        l_ = np.ndarray(shape=(len(ID), (2 * corePadSize + 1) ** 2, 2),
                        dtype=np.float32)
        im_ = np.ndarray(shape=(len(ID), WindowSize[0], WindowSize[1], WindowSize[2],
                                WindowSize[3]), dtype=np.float32)
        for i in range(len(ID)):
            r = np.unravel_index(ID[i], l.shape)
            im_[i, :, :, :] = im[r[0]:(r[0] + WindowSize[0]),
                              r[1]:(r[1] + WindowSize[1]), r[2]:(r[2] + WindowSize[2]), :]
            l_[i, :, 1] = np.reshape(l[r[0],
                                     (r[1] - corePadSize):(r[1] + corePadSize + 1),
                                     (r[2] - corePadSize):(r[2] + corePadSize + 1), :],
                                     (2 * corePadSize + 1) ** 2)
            l_[i, :, 0] = 1 - l_[i, :, 1]
        return im_, l_

    def get_batch3d_fwd(im, Vshape, ID):
        """ generate a batch from im for testing
            based on the location of ID entries and core pad size. Note that the ID
            is based on no core pad.
        """
        im_ = np.ndarray(shape=(len(ID), WindowSize[0], WindowSize[1], WindowSize[2]
                                , WindowSize[3]), dtype=np.float32)
        for i in range(len(ID)):
            r = np.unravel_index(ID[i], Vshape)
            im_[i, :, :, :] = im[r[0]:r[0] + WindowSize[0], r[1]:r[1] + WindowSize[1],
                              r[2]:r[2] + WindowSize[2], r[3]:r[3] + WindowSize[3]]
        return im_

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
    y_conv = tf.reshape(h_fc1, [-1, 1 * 5 * 5, 2])

    # loss function over (TP U FN U FP)
    allButTN = tf.maximum(tf.argmax(y_conv, 2), tf.argmax(y_, 2))
    cross_entropy = tf.reduce_mean(tf.multiply(tf.cast(allButTN, tf.float32),
                                               tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    correct_prediction = tf.multiply(tf.argmax(y_conv, 2), tf.argmax(y_, 2))
    accuracy = tf.divide(tf.reduce_sum(tf.cast(correct_prediction, tf.float32)),
                         tf.reduce_sum(tf.cast(allButTN, tf.float32)))
   
    sess.run(tf.global_variables_initializer())
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    if isForward:
        saver.restore(sess, os.path.join(os.path.dirname(os.path.realpath(__file__)), "private/model-epoch29999.ckpt"))
        print("Model restored.")
        vID = []
        for ii in range(0, V.shape[0]):
            for ij in it.chain(range(corePadSize, V.shape[1] - corePadSize,2 * corePadSize + 1), [V.shape[1] - corePadSize - 1]):
                for ik in it.chain(range(corePadSize, V.shape[2] - corePadSize, 2 * corePadSize + 1), [V.shape[2] - corePadSize - 1]):
                    vID.append(np.ravel_multi_index((ii, ij, ik, 0), V.shape))

        steps_num = len(vID)
        print(steps_num, 'steps_num')
        start = time.time()
        for i in range(np.int(np.ceil(len(vID) / batch_size))):
            x1 = get_batch3d_fwd(im, imShape, vID[i * batch_size:(i + 1) * batch_size])
            y1 = np.reshape(y_conv.eval(feed_dict={x: x1, keep_prob: 1.0}), (-1,
                                                                             (2 * corePadSize + 1),
                                                                             (2 * corePadSize + 1), 2))
            for j in range(y1.shape[0]):
                r = np.unravel_index(vID[i * batch_size + j], V.shape)
                V[r[0], (r[1] - corePadSize):(r[1] + corePadSize + 1),
                (r[2] - corePadSize):(r[2] + corePadSize + 1), 0] = np.argmax(y1[j], axis=2)

            if i % 100 == 99:
                end = time.time()
                print("step %d is done.  %f min to finish." % (i,
                                                               (end - start)/ 60 / (i + 1) *
                                                               (np.int(np.ceil(len(vID) / batch_size)) - i - 1)))

        io.savemat(inputData[:-3] + '-V_fwd', {'V':
                                                   np.transpose(np.reshape(V, imShape[0:3]), (2, 1, 0))})
    print(inputData[:-3] + "V_fwd.mat is saved.")

if __name__ == '__main__':
    print('Start tracing model')
    inputData = ''
    if len(sys.argv) > 1:
        inputData = sys.argv[1]
    start_tracing_model(inputData)
