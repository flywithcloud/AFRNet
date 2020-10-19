import tensorflow as tf
import numpy as np
import sys
import os, os.path
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import util

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def classificationLayer(x,classes,name="classification2",reuse=False,isTrainable=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=classes,
                              kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),
                              activation=None, name='fc1', trainable=isTrainable,reuse=reuse)
        net = tf.reshape(net, [-1, classes])
    return net


class CLASSIFICATION2:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, modeldir, _lr, _beta1=0.5, _nepoch=20, _batch_size=100):
        self.train_X = _train_X  # feature of trainset
        self.train_Y = _train_Y  # label of trainset

        self.test_seen_feature = data_loader.test_seen_feature  # feature of seen classes
        self.test_seen_label = data_loader.test_seen_label   # label of seen classes
        self.test_unseen_feature = data_loader.test_unseen_feature  # feature of unseen classes
        self.test_unseen_label = data_loader.test_unseen_label  # label of unseen classes

        self.seenclasses = data_loader.seenclasses  # seen classes
        self.unseenclasses = data_loader.unseenclasses  # unseen classes
        self.nclass = _nclass  # number of all classes
        self.input_dim = _train_X.shape[1]  # dim of visual feature
        self.ntrain = self.train_X.shape[0]  # number of trainset feature

        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.lr = _lr
        self.beta1 = _beta1
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.modeldir = modeldir

        # model_definition
        self.input = tf.placeholder(tf.float32, [None, self.input_dim], name='input')
        self.label = tf.placeholder(tf.int32, [None], name='label')

        self.classificationLogits = classificationLayer(self.input, self.nclass)

        # classification loss
        self.classificationLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classificationLogits, labels=self.label))
        classifierParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classification2')

        for params in classifierParams:
            print (params.name)
        print ('...................')

        classifierOptimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.beta1,beta2=0.99)
        classifierGradsVars = classifierOptimizer.compute_gradients(self.classificationLoss,var_list=classifierParams)    
        self.classifierTrain = classifierOptimizer.apply_gradients(classifierGradsVars)

        self.saver = tf.train.Saver()

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.nepoch):
                self.ntrain = self.train_X.shape[0]  # number of trainset feature
                for i in range(0, self.ntrain, self.batch_size):
                    # train classifier
                    batch_input, batch_label = self.next_batch(self.batch_size)
                    _,loss = sess.run([self.classifierTrain, self.classificationLoss],
                                             feed_dict={self.input:batch_input,self.label:batch_label})
                    # print ("Classification loss is:"+str(loss))

                # test on seen classes
                start = 0
                ntest = self.test_seen_feature.shape[0]
                predicted_label = np.empty_like(self.test_seen_label)
                for i in range(0, ntest, self.batch_size):
                    end = min(ntest, start + self.batch_size)
                    output = sess.run([self.classificationLogits], feed_dict={self.input: self.test_seen_feature[start:end]})
                    predicted_label[start:end] = np.argmax(np.squeeze(np.array(output)), axis=1)
                    start = end
                acc_seen = self.compute_per_class_acc_gzsl(self.test_seen_label, predicted_label, self.seenclasses)*100

                # test on unseen classes
                start = 0
                ntest = self.test_unseen_feature.shape[0]
                predicted_label = np.empty_like(self.test_unseen_label)
                for i in range(0, ntest, self.batch_size):
                    end = min(ntest, start + self.batch_size)
                    output = sess.run([self.classificationLogits],
                                      feed_dict={self.input: self.test_unseen_feature[start:end]})
                    predicted_label[start:end] = np.argmax(np.squeeze(np.array(output)), axis=1)
                    start = end
                acc_unseen = self.compute_per_class_acc_gzsl(self.test_unseen_label, predicted_label, self.unseenclasses)*100

                H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
                print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))

                if H > best_H:
                    best_seen = acc_seen
                    best_unseen = acc_unseen
                    best_H = H
                    # self.saver.save(sess, os.path.join(self.modeldir, 'models_best.ckpt'))
                    # print ("Model saved")

        return best_seen, best_unseen, best_H

    def fit_zsl(self):
        best_acc = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.nepoch):
                self.ntrain = self.train_X.shape[0]  # number of trainset feature
                for i in range(0, self.ntrain, self.batch_size):
                    # train classifier
                    batch_input, batch_label = self.next_batch(self.batch_size)
                    _,loss = sess.run([self.classifierTrain,self.classificationLoss],
                                             feed_dict={self.input:batch_input,self.label:batch_label})

                # test on unseen classes
                start = 0
                ntest = self.test_unseen_feature.shape[0]
                predicted_label = np.empty_like(self.test_unseen_label)
                for i in range(0, ntest, self.batch_size):
                    end = min(ntest, start + self.batch_size)
                    output = sess.run([self.classificationLogits], feed_dict={self.input: self.test_unseen_feature[start:end]})
                    predicted_label[start:end] = np.argmax(np.squeeze(np.array(output)), axis=1)
                    start = end
                acc = self.compute_per_class_acc(util.map_label(self.test_unseen_label, self.unseenclasses), predicted_label, self.unseenclasses.shape[0])*100
                print ('accuracy is', acc)

                if acc > best_acc:
                    best_acc = acc
                    # print(best_acc)
                    # self.saver.save(sess, os.path.join(self.modeldir, 'models_best.ckpt'))
                    # print ("Model saved")
        return best_acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0.0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += np.sum(test_label[idx]==predicted_label[idx])*1.0 / np.sum(idx)
        acc_per_class /= target_classes.shape[0]
        return acc_per_class

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = np.zeros(nclass)
        for i in range(0, nclass):
            idx = (test_label == i)
            acc_per_class[i] = np.sum(test_label[idx]==predicted_label[idx])*1.0 / np.sum(idx)
        return np.mean(acc_per_class)

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.ntrain = self.train_X.shape[0]  # number of trainset feature
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = np.random.permutation(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = np.random.permutation(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return np.concatenate((X_rest_part, X_new_part), axis=0) , np.concatenate((Y_rest_part, Y_new_part), axis=0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]