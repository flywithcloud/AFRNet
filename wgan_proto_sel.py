from __future__ import print_function
import os, os.path
import tensorflow as tf
import sys
import random
import argparse
import util
import numpy as np
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


def discriminator(x, opt, name="discriminator", reuse=False, isTrainable=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=opt.ndh,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                              activation=leaky_relu, name='disc_fc1', trainable=isTrainable, reuse=reuse)

        real_fake = tf.layers.dense(inputs=net, units=1,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                                    activation=None, name='disc_rf', trainable=isTrainable, reuse=reuse)
        return tf.reshape(real_fake, [-1])


def next_feed_dict(data, opt):
    batch_feature, batch_label, batch_att, batch_proto = data.next_batch(opt.batch_size)
    z_rand = np.random.normal(0, 1, [opt.batch_size, opt.nz]).astype(np.float32)
    return batch_feature, batch_att, batch_label, batch_proto, z_rand


def wgan_proto_sel(opt, data):
    # graph1 definition
    g1 = tf.Graph()
    with g1.as_default():
        # placeholders
        input_res = tf.placeholder(tf.float32, [None, opt.resSize], name='input_features')
        input_proto = tf.placeholder(tf.float32, [None, opt.resSize], name='input_feature_prototypes')
        input_att = tf.placeholder(tf.float32, [None, opt.attSize], name='input_attributes')
        noise_z = tf.placeholder(tf.float32, [None, opt.nz], name='noise')
        input_label = tf.placeholder(tf.int32, [None], name='input_label')

        # model definition
        train = True
        reuse = False

        # input of generator
        noise = tf.concat([noise_z, input_att], axis=1)

        # output of generator
        gen_res = generator(noise, opt, isTrainable=train, reuse=reuse)
        gen_res = gen_res + input_proto

        # input of discriminator
        targetEmbd = tf.concat([input_res, input_att], axis=1)
        genTargetEmbd = tf.concat([gen_res, input_att], axis=1)

        # output of discriminator
        targetDisc = discriminator(targetEmbd, opt,isTrainable=train,reuse=reuse)
        genTargetDisc = discriminator(genTargetEmbd, opt,isTrainable=train, reuse=True)

        # discriminator loss
        discriminatorLoss = tf.reduce_mean(genTargetDisc - targetDisc)
        alpha = tf.random_uniform(shape=[opt.batch_size, 1], minval=0., maxval=1.)
        interpolates = alpha * input_res + ((1 - alpha) * gen_res)
        interpolate = tf.concat([interpolates, input_att], axis=1)
        interpolate_disc = discriminator(interpolate, opt, reuse=True,isTrainable=train)
        gradients = tf.gradients(interpolate_disc, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradientPenalty = tf.reduce_mean((slopes - 1.) ** 2)
        gradientPenalty = opt.lambda1 * gradientPenalty
        discriminatorLoss = discriminatorLoss + gradientPenalty

        # Wasserstein generator loss
        genDiscMean = tf.reduce_mean(genTargetDisc)
        genLoss = -genDiscMean
        generatorLoss = genLoss

        # getting parameters to optimize
        discParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        generatorParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        for params in discParams:
            print(params.name)
        print('...................')

        for params in generatorParams:
            print(params.name)
        print('...................')

        discOptimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=0.9)
        genOptimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=0.9)

        discGradsVars = discOptimizer.compute_gradients(discriminatorLoss, var_list=discParams)
        genGradsVars = genOptimizer.compute_gradients(generatorLoss, var_list=generatorParams)

        discTrain = discOptimizer.apply_gradients(discGradsVars)
        generatorTrain = genOptimizer.apply_gradients(genGradsVars)

    # training g1 graph
    with tf.Session(graph=g1, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)

        best_acc = 0
        syn_res = np.empty((0, opt.resSize), np.float32)
        syn_label = np.empty((0), np.float32)
        classifier_zsl = classifier.CLASSIFICATION2(syn_res, syn_label, data,
                                                     data.ntest_class, opt.modeldir + '_classifier_zsl', opt.lr_cls,
                                                     0.5, opt.nepoch_cls, opt.bs_cls)
        for epoch in range(opt.nepoch):
            # train discriminator and generator
            for i in range(0, data.ntrain, opt.batch_size):
                for j in range(opt.critic_iter):
                    # train discriminator
                    batch_feature, batch_att, batch_label, batch_proto, z_rand = next_feed_dict(data, opt)
                    _, discLoss = sess.run([discTrain, discriminatorLoss],
                                           feed_dict={input_res: batch_feature, input_att: batch_att,
                                                      input_label: batch_label, input_proto:batch_proto, noise_z: z_rand})
                    print("Discriminator loss is:" + str(discLoss))

                # train generator
                batch_feature, batch_att, batch_label, batch_proto, z_rand = next_feed_dict(data, opt)
                _, genLoss = sess.run([generatorTrain, generatorLoss],
                                              feed_dict={input_att: batch_att, input_proto: batch_proto,
                                                         input_label: batch_label, noise_z: z_rand})
                print('epoch:', epoch, "Generator loss is:" + str(genLoss))

            # synthesis feature of unseen classes
            syn_res = np.empty((0, opt.resSize), np.float32)
            syn_label = np.empty((0), np.float32)
            for i,c in enumerate(data.unseenclasses):
                iclass_att = np.reshape(data.attribute[c], (1, opt.attSize))
                iclass_proto = np.reshape(data.test_unseen_proto[i],(1, opt.resSize))

                batch_att = np.repeat(iclass_att, [opt.syn_num], axis=0)
                batch_proto = np.repeat(iclass_proto, [opt.syn_num], axis=0)
                z_rand = np.random.normal(0, 1, [opt.syn_num, opt.nz]).astype(np.float32)

                syn_features = sess.run(gen_res, feed_dict={input_att: batch_att, input_proto:batch_proto, noise_z: z_rand})
                syn_res = np.vstack((syn_res, syn_features))
                temp = np.repeat(i, [opt.syn_num], axis=0)
                syn_label = np.concatenate((syn_label, temp))
            # shuffle the data
            idx = range(0, syn_label.shape[0])
            shuffle(idx)
            syn_res = syn_res[idx, :]
            syn_label = syn_label[idx]

            # generated feature evaluation
            # train model
            classifier_zsl.train_X = syn_res
            classifier_zsl.train_Y = syn_label

            acc_tr = classifier_zsl.fit_zsl()
            print('ZSL training: unseen class accuracy= ', acc_tr)

            if acc_tr > best_acc:
                best_acc = acc_tr
                saver.save(sess, os.path.join(opt.modeldir, 'best_acc.ckpt'))

        print('best_acc', best_acc)


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
    parser.add_argument('--preprocessing', default=True, help='enbale MinMaxScaler on visual features')

    # about architecture
    parser.add_argument('--nz', type=int, default=64, help='size of the noise latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')

    # about training wgan
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train GAN')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate to train GANs ')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')

    parser.add_argument('--modeldir', default='./APY_proto1024_model', help='folder to output  model checkpoints')

    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--val_interval', type=int, default=1, help='interval to test the accuracy of classification')

    # generate feature
    parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')

    # evaluate feature
    parser.add_argument('--nepoch_cls', type=int, default=40, help='number of epochs to train cls')
    parser.add_argument('--bs_cls', type=int, default=128, help='number features to generate per class')
    parser.add_argument('--lr_cls', type=float, default=0.0002, help='learning rate to train classifier ')

    opt = parser.parse_args()
    print(opt)

    # set random seed
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

    # train GAN
    wgan_proto_sel(opt, data)

