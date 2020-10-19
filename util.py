from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn import preprocessing


def map_label(label, classes):
    mapped_label = np.empty_like(label)
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i]] = i
    return mapped_label


class DATA_LOADER_proto(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.shuffle_data_idx()

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        train_proto = np.load(opt.dataroot + '/' + opt.dataset + '/pre_seen_exemplar2048.npy')
        test_unseen_proto = np.load(opt.dataroot + '/' + opt.dataset + '/pre_unseen_exemplar2048.npy')

        self.attribute = matcontent['att'].T.astype(np.float32)  # attribute

        if opt.preprocessing:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            _test_unseen_proto = scaler.transform(test_unseen_proto)
            _train_proto = scaler.transform(train_proto)

            self.train_feature = _train_feature.astype(np.float32)
            self.train_label = label[trainval_loc].astype(np.int)

            self.test_unseen_feature = _test_unseen_feature.astype(np.float32)
            self.test_unseen_label = label[test_unseen_loc].astype(np.int)

            self.test_seen_feature = _test_seen_feature.astype(np.float32)
            self.test_seen_label = label[test_seen_loc].astype(np.int)

            self.train_proto = _train_proto.astype(np.float32)
            self.test_unseen_proto = _test_unseen_proto.astype(np.float32)

        else:
            self.train_feature = feature[trainval_loc].astype(np.float32)
            self.train_label = label[trainval_loc].astype(np.int)

            self.test_unseen_feature = feature[test_unseen_loc].astype(np.float32)
            self.test_unseen_label = label[test_unseen_loc].astype(np.int)

            self.test_seen_feature = feature[test_seen_loc].astype(np.float32)
            self.test_seen_label = label[test_seen_loc].astype(np.int)

            self.train_proto = train_proto.astype(np.float32)
            self.test_unseen_proto = test_unseen_proto.astype(np.float32)

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)

        self.ntrain = (self.train_feature).shape[0]
        self.ntrain_class = (self.seenclasses).shape[0]
        self.ntest_class = (self.unseenclasses).shape[0]
        self.train_class = np.copy(self.seenclasses)
        self.allclasses = np.arange(0, self.ntrain_class + self.ntest_class).astype(np.int)

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)  # from 0 to num_train_classes
        self.test_unseen_mapped_label = map_label(self.test_unseen_label, self.unseenclasses)

    def shuffle_data_idx(self):
        """randomly permute the training data"""
        self.perm = np.random.permutation(np.arange(self.ntrain))
        self.cur = 0

    def next_batch(self, batch_size):
        """return the train data for the next batch"""
        if self.cur + batch_size >= self.ntrain:
            self.shuffle_data_idx()
        data_idx = self.perm[self.cur: self.cur + batch_size]
        self.cur += batch_size

        batch_feature = self.train_feature[data_idx]
        batch_label = self.train_label[data_idx]
        batch_att = self.attribute[batch_label]
        batch_label_map = self.train_mapped_label[data_idx]
        batch_proto = self.train_proto[batch_label_map]
        return batch_feature, batch_label_map, batch_att, batch_proto


class DATA_LOADER_sel(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.shuffle_data_idx()

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        _train_feature = feature[trainval_loc]
        _test_seen_feature = feature[test_seen_loc]
        _test_unseen_feature = feature[test_unseen_loc]

        _train_proto = np.load(opt.dataroot + '/' + opt.dataset + '/pre_seen_exemplar2048.npy')
        _test_unseen_proto = np.load(opt.dataroot + '/' + opt.dataset + '/pre_unseen_exemplar2048.npy')

        if opt.select:
            pre_proto_mse = np.load(opt.dataroot + '/' + opt.dataset + '/pre_seen_exemplar_mse.npy')
            pre_proto_mse_sort = np.argsort(pre_proto_mse)[:opt.resSize]  # index of dim with the smallest mse
            _train_feature = _train_feature[:, pre_proto_mse_sort]
            _test_unseen_feature = _test_unseen_feature[:, pre_proto_mse_sort]
            _test_seen_feature = _test_seen_feature[:, pre_proto_mse_sort]
            _train_proto = _train_proto[:, pre_proto_mse_sort]
            _test_unseen_proto = _test_unseen_proto[:, pre_proto_mse_sort]

        self.attribute = matcontent['att'].T.astype(np.float32)  # attribute

        if opt.preprocessing:
            scaler = preprocessing.MinMaxScaler()

            _train_feature = scaler.fit_transform(_train_feature)
            _test_seen_feature = scaler.transform(_test_seen_feature)
            _test_unseen_feature = scaler.transform(_test_unseen_feature)
            _test_unseen_proto = scaler.transform(_test_unseen_proto)
            _train_proto = scaler.transform(_train_proto)

            self.train_feature = _train_feature.astype(np.float32)
            self.train_label = label[trainval_loc].astype(np.int)

            self.test_unseen_feature = _test_unseen_feature.astype(np.float32)
            self.test_unseen_label = label[test_unseen_loc].astype(np.int)

            self.test_seen_feature = _test_seen_feature.astype(np.float32)
            self.test_seen_label = label[test_seen_loc].astype(np.int)

            self.train_proto = _train_proto.astype(np.float32)
            self.test_unseen_proto = _test_unseen_proto.astype(np.float32)

        else:
            self.train_feature = feature[trainval_loc].astype(np.float32)
            self.train_label = label[trainval_loc].astype(np.int)

            self.test_unseen_feature = feature[test_unseen_loc].astype(np.float32)
            self.test_unseen_label = label[test_unseen_loc].astype(np.int)

            self.test_seen_feature = feature[test_seen_loc].astype(np.float32)
            self.test_seen_label = label[test_seen_loc].astype(np.int)

            self.train_proto = _train_proto.astype(np.float32)
            self.test_unseen_proto = _test_unseen_proto.astype(np.float32)

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)

        self.ntrain = (self.train_feature).shape[0]
        self.ntrain_class = (self.seenclasses).shape[0]
        self.ntest_class = (self.unseenclasses).shape[0]
        self.train_class = np.copy(self.seenclasses)
        self.allclasses = np.arange(0, self.ntrain_class + self.ntest_class).astype(np.int)

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)  # from 0 to num_train_classes
        self.test_unseen_mapped_label = map_label(self.test_unseen_label, self.unseenclasses)

    def shuffle_data_idx(self):
        """randomly permute the training data"""
        self.perm = np.random.permutation(np.arange(self.ntrain))
        self.cur = 0

    def next_batch(self, batch_size):
        """return the train data for the next batch"""
        if self.cur + batch_size >= self.ntrain:
            self.shuffle_data_idx()
        data_idx = self.perm[self.cur: self.cur + batch_size]
        self.cur += batch_size

        batch_feature = self.train_feature[data_idx]
        batch_label = self.train_label[data_idx]
        batch_att = self.attribute[batch_label]
        batch_label_map = self.train_mapped_label[data_idx]
        batch_proto = self.train_proto[batch_label_map]
        return batch_feature, batch_label_map, batch_att, batch_proto