from __future__ import print_function
import os, os.path
import numpy as np
import tensorflow as tf
import sys
import random
import argparse
import util
from random import shuffle
import classifier

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def leaky_relu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def generator(x, opt, name="generator", reuse=False, isTrainable=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=opt.ngh,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                              activation=leaky_relu, name='gen_fc1', trainable=isTrainable, reuse=reuse)

        net = tf.layers.dense(inputs=net, units=opt.resSize,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                              activation=tf.nn.relu, name='gen_fc2', trainable=isTrainable, reuse=reuse)
        return tf.reshape(net, [-1, opt.resSize])


def eval_proto_feat(data, opt):
    # graph 2 definition
    g2 = tf.Graph()
    with g2.as_default():
        # model definition
        input_proto = tf.placeholder(tf.float32, [opt.syn_num, opt.resSize], name='input_feature_prototypes')

        syn_att = tf.placeholder(tf.float32, [opt.syn_num, opt.attSize], name='input_attributes')
        noise_z1 = tf.placeholder(tf.float32, [opt.syn_num, opt.nz], name='noise')
        noise1 = tf.concat([noise_z1, syn_att], axis=1)

        gen_res = generator(noise1, opt, isTrainable=False, reuse=False)
        gen_res = gen_res + input_proto

    # getting features from g2 graph
    with tf.Session(graph=g2, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        for var in params:
            print (var.name + "\t")
        saver = tf.train.Saver(var_list=params)
        string = opt.modeldir + '/best_acc.ckpt'
        print (string)
        try:
            saver.restore(sess, string)
        except:
            print("Previous weights not found of generator")
            sys.exit(0)
        print ("Model loaded")

        # synthesis feature of unseen classes
        syn_res = np.empty((0, opt.resSize), np.float32)
        syn_label = np.empty((0), np.float32)
        for i, c in enumerate(data.unseenclasses):
            iclass_att = np.reshape(data.attribute[c], (1, opt.attSize))
            iclass_proto = np.reshape(data.test_unseen_proto[i], (1, opt.resSize))

            batch_att = np.repeat(iclass_att, [opt.syn_num], axis=0)
            batch_proto = np.repeat(iclass_proto, [opt.syn_num], axis=0)
            z_rand = np.random.normal(0, 1, [opt.syn_num, opt.nz]).astype(np.float32)

            syn_features = sess.run(gen_res, feed_dict={syn_att: batch_att, input_proto: batch_proto, noise_z1: z_rand})
            syn_res = np.vstack((syn_res, syn_features))
            temp = np.repeat(i, [opt.syn_num], axis=0)
            syn_label = np.concatenate((syn_label, temp))
        # shuffle the data
        idx = range(0, syn_label.shape[0])
        shuffle(idx)
        syn_res = syn_res[idx, :]
        syn_label = syn_label[idx]
        print (syn_res.shape)
        print (syn_label.shape)

        # define model for zsl
        classifier_zsl = classifier.CLASSIFICATION2(syn_res, syn_label, data,
                                                         data.ntest_class, opt.modeldir + '_classifier_zsl', opt.lr,
                                                         0.5, opt.nepoch, opt.bs)
        # train model
        acc = classifier_zsl.fit_zsl()
        print('unseen class accuracy= ', acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # about data
    parser.add_argument('--dataset', default='APY', help='AWA1')
    parser.add_argument('--dataroot', default='../data/xlsa17/data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')

    parser.add_argument('--select', type=bool, default=True)
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=64, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')
    parser.add_argument('--preprocessing', default=True, help='enbale MinMaxScaler on visual features')

    parser.add_argument('--modeldir', default='./APY_proto1024_model', help='folder to output  model checkpoints')
    parser.add_argument('--nepoch', type=int, default=40, help='number of epochs to train classifier')
    parser.add_argument('--bs', type=int, default=128, help='number features to generate per class')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate to train classifier ')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

    opt = parser.parse_args()
    print(opt)

    random.seed(opt.manualSeed)
    tf.set_random_seed(opt.manualSeed)

    if not os.path.exists(opt.modeldir):
        os.makedirs(opt.modeldir)

    # data reading
    data = util.DATA_LOADER_sel(opt)
    print("#####################################")
    print("# of training samples: ", data.ntrain)
    print(data.seenclasses)
    print(data.unseenclasses)
    print(data.ntrain_class)
    print(data.ntest_class)
    print("#####################################")

    eval_proto_feat(data, opt)