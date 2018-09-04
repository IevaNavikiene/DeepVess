from __future__ import print_function
import numpy as np
import time
from random import shuffle

def train_deep_vess(trnL, it, corePadSize, tstL, tst, nEpoch, trn, train_step, accuracy, keep_prob, x, y_, saver, sess, WindowSize):
    def get_batch(im, l, corePadSize, ID):
        """ generate a batch from im and l for training

            based on the location of ID entries and core pad size. Note that the ID
            is based on no core pad.
        """
        l_ = np.ndarray(shape=(len(ID), (2 * corePadSize + 1) ** 2, 2),
                        dtype=np.float32)
        im_ = np.ndarray(shape=(len(ID), WindowSize[0], WindowSize[1], WindowSize[2],
                                WindowSize[3]), dtype=np.float32)
        for i in xrange(len(ID)):
            r = np.unravel_index(ID[i], l.shape)
            im_[i, :, :, :] = im[r[0]:(r[0] + WindowSize[0]),
                              r[1]:(r[1] + WindowSize[1]), r[2]:(r[2] + WindowSize[2]), :]
            l_[i, :, 1] = np.reshape(l[r[0],
                                     (r[1] - corePadSize):(r[1] + corePadSize + 1),
                                     (r[2] - corePadSize):(r[2] + corePadSize + 1), :],
                                     (2 * corePadSize + 1) ** 2)
            l_[i, :, 0] = 1 - l_[i, :, 1]
        return im_, l_


    file_log = open("model.log", "w")
    file_log.write("Epoch, Step, training accuracy, test accuracy, Time (hr) \n")
    file_log.close()
    start = time.time()
    begin = start
    trnSampleID = []
    for ii in xrange(0, trnL.shape[0]):
        for ij in it.chain(xrange(corePadSize,
                    trnL.shape[1] - corePadSize, 2 * corePadSize + 1),
                    [trnL.shape[1] - corePadSize - 1]):
            for ik in it.chain(xrange(corePadSize,trnL.shape[2]-corePadSize,
                    2*corePadSize + 1), [trnL.shape[2] - corePadSize - 1]):
                trnSampleID.append(np.ravel_multi_index((ii, ij, ik, 0),
                                                        trnL.shape))
    shuffle(trnSampleID)
    tstSampleID = []
    for ii in xrange(0, tstL.shape[0]):
        for ij in it.chain(xrange(corePadSize, tstL.shape[1] - corePadSize,
                     2 * corePadSize + 1), [tstL.shape[1] - corePadSize - 1]):
            for ik in it.chain(xrange(corePadSize, tstL.shape[2] - corePadSize,
                     2 * corePadSize + 1), [tstL.shape[2] - corePadSize - 1]):
                tstSampleID.append(np.ravel_multi_index((ii, ij, ik, 0),
                                                        tstL.shape))
    shuffle(tstSampleID)
    x_tst,l_tst = get_batch(tst, tstL, corePadSize, tstSampleID[0:1000])
    for epoch in xrange(nEpoch):
        shuffle(trnSampleID)
        for i in xrange(np.int(np.ceil(len(trnSampleID) / 1000.))):
          x1,l1 = get_batch(trn, trnL, corePadSize,
                            trnSampleID[(i * 1000):((i + 1) * 1000)])
          train_step.run(feed_dict={x: x1, y_: l1, keep_prob: 0.5})
          if i%100 == 99:
            train_accuracy = accuracy.eval(feed_dict={
                x: x1 , y_: l1 , keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={
                x: x_tst , y_: l_tst, keep_prob: 1.0})
            end = time.time()
            print("epoch %d, step %d, training accuracy %g, test accuracy %g. "
                "Elapsed time/sample is %e sec. %f hour to finish."%(epoch, i,
                train_accuracy, test_accuracy, (end - start) / 100000,
                ((nEpoch - epoch) * len(trnSampleID) / 1000 - i)
                * (end - start) / 360000))
            file_log = open("model.log","a")
            file_log.write("%d, %d, %g, %g, %f \n" % (epoch, i, train_accuracy,
                                         test_accuracy, (end-begin) / 3600))
            file_log.close()
            start = time.time()
        if epoch%10 == 9:
            save_path = saver.save(sess, "model-epoch" + str(epoch) + ".ckpt")
            print("epoch %d, Model saved in file: %s" % (epoch, save_path))
