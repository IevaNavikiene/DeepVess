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
from api.resources.preprocessing.DeepVess.TrainDeepVess import train_deep_vess
from api.resources.preprocessing.DeepVess.DeepVessModel import define_deepvess_architecture

def start_tracing_model(inputData, isTrain=False, isForward = True, padSize = ((3, 3), (16, 16), (16, 16), (0, 0)), nEpoch = 100):
    """

    :param inputData:
    :param isTrain: Change isTrain to True if you want to train the network
    :param isForward:  Change isForward to True if you want to test the network
    :param padSize: padSize is the padding around the central voxel to generate the field of view
    :param nEpoch: number of epoch to train
    :return:
    """

    WindowSize = np.sum(padSize, axis=1) + 1
    # pad Size aroung the central voxel to generate 2D region of interest
    corePadSize = 2
    keep_prob = 1.0
    # start the TF session
    sess = tf.InteractiveSession()
    #create placeholder for input and output nodes
    x = tf.placeholder(tf.float32, shape=[None, WindowSize[0], WindowSize[1],
                            WindowSize[2], WindowSize[3]])
    y_ = tf.placeholder(tf.float32, shape=[None, (2 * corePadSize + 1) ** 2, 2])

    # Import Data
    f = h5py.File(inputData, 'r')
    im = np.array(f.get('/im'))
    im = im.reshape(im.shape + (1, ))
    imShape = im.shape

    if isTrain:
        l = np.array(f.get('/l'))
        l = l.reshape(l.shape + (1,))
        nc = im.shape[1]
        tst = im[:, (nc / 2):(3 * nc / 4), :]
        tstL = l[:,(nc / 2):(3 * nc / 4), :]
        trn = im[:, 0:(nc / 2), :]
        trnL = l[:, 0:(nc / 2), :]
        tst = np.pad(tst, padSize, 'symmetric')
        trn = np.pad(trn, padSize, 'symmetric')
    if isForward:
        im = np.pad(im, padSize, 'symmetric')
        V = np.ndarray(shape=(im.shape), dtype=np.float32)
    print("Data loaded.")

    def get_batch3d_fwd(im, Vshape, ID):
        """
          generate a batch from im for testing

          based on the location of ID entries and core pad size. Note that the ID
          is based on no core pad.
        """
        im_ = np.ndarray(shape=(ID.size, WindowSize[0], WindowSize[1], WindowSize[2], WindowSize[3]), dtype=np.float32)
        for i in range(ID.size):
            r = np.unravel_index(ID,Vshape)
            x = 0
            y = 0
            im_[i, :, :, :] = im[r[0]:(r[0] + WindowSize[0]),
                (r[1] + y):(r[1] + WindowSize[1] + y),
                (r[2] + x):(r[2] + WindowSize[2] + x), r[3]:(r[3] + WindowSize[3])]
        return im_


    y_conv, keep_prob_tensor = define_deepvess_architecture(x)
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

    if isTrain:
        train_deep_vess(trnL, it, corePadSize, tstL, tst, nEpoch, trn, train_step, accuracy, keep_prob, x, y_, saver, sess, WindowSize)
        
    if isForward:
        saver.restore(sess,os.path.join(os.path.dirname(os.path.realpath(__file__)), "private/model-epoch29999.ckpt"))
        print("Model restored.")
        vID=[]


        for ii in range(0,V.shape[0]):
            for ij in it.chain(range(corePadSize, V.shape[1] - corePadSize,
                            2 * corePadSize + 1), [V.shape[1] - corePadSize - 1]):
                for ik in it.chain(range(corePadSize, V.shape[2] - corePadSize,
                            2 * corePadSize + 1), [V.shape[2] - corePadSize - 1]):
                    vID.append(np.ravel_multi_index((ii, ij, ik, 0), V.shape))

        steps_num = len(vID)
        print(steps_num, 'steps_num')
        for i in vID:
            x1 = get_batch3d_fwd(im, imShape, np.array(i))
            y1 = np.reshape(y_conv.eval(feed_dict={x: x1, keep_prob_tensor: keep_prob}),
                          ((2 * corePadSize + 1), (2 * corePadSize + 1), 2))
            r = np.unravel_index(i, V.shape)
            V[r[0], (r[1] - corePadSize):(r[1] + corePadSize + 1),
                (r[2] - corePadSize):(r[2] + corePadSize + 1), 0] = np.argmax(y1,
                                                                        axis=2)
            if i%10000 == 9999:
                #if i%2 == 0:
                print("step %d percent is done. i:%d , steps_num: %d" % (i/ steps_num,i,steps_num))
                break
        io.savemat(inputData[:-3] + '-V_fwd',{'V':
            np.transpose(np.reshape(V, imShape[0:3]), (2, 1, 0))})
        print(inputData[:-3] + "V_fwd.mat is saved.")
    
if __name__ == '__main__':
    print('Start tracing model')
    inputData = ''
    if len(sys.argv) > 1:
        inputData = sys.argv[1]
    start_tracing_model(inputData)
