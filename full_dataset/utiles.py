import numpy as np
import h5py
import torch
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler
import random

def create_label():
#    mods = [0'32PSK', 1'16APSK', 2'32QAM', 3'FM', 4'GMSK', 5'32APSK', 6'OQPSK', 7'8ASK', 8'BPSK', 9'8PSK', 10'AM-SSB-SC',
#            11'4ASK', 12'16PSK', 13'64APSK', 14'128QAM', 15'128APSK', 16'AM-DSB-SC', 17'AM-SSB-WC', 18'64QAM', 19'QPSK',
#            20'256QAM', 21'AM-DSB-WC', 22'OOK', 23'16QAM']

    mo = []
    for m in range(24):
        mo.append([m] * 4096 * 26)

    mo = np.expand_dims(np.hstack(mo), axis=1)
    print (np.shape(mo))

    return mo


def gen_list():

    labels = create_label()
    n_examples = labels.shape[0]
    test_idx = []
    for n in range(24 * 26):
        per_slot = np.random.choice(range(n * 4096, (n + 1) * 4096), size=int(512), replace=False)
        test_idx.extend(per_slot)

    train_idx0 = list(set(range(0, n_examples)) - set(test_idx))
    train_idx = random.sample(train_idx0, k = len(train_idx0))

    return test_idx, train_idx


def gen_test_set(dataset, test_idx0):

    x = h5py.File(dataset, 'r+')
    labels = create_label()

    test_idx = random.sample(test_idx0, k = int(len(test_idx0) * 0.5))

    data_test0 = []
    for it in test_idx:
        data_test0.append(x['X'][it, :, :])
    
    data_test0 = np.asarray(data_test0)
    label_test0 = labels[test_idx, :]

    data_test = torch.from_numpy(data_test0).float()
    label_test = torch.from_numpy(label_test0).long()

    test_set = utils.TensorDataset(data_test, label_test)

    return test_set


def gen_train_set(dataset, train_idx):
    
    x = h5py.File(dataset, 'r+')
    labels = create_label()

    data_train0 = []
    for it in train_idx:
        data_train0.append(x['X'][it, :, :])

    data_train0 = np.asarray(data_train0)
    label_train0 = labels[train_idx, :]

    data_train = torch.from_numpy(data_train0).float()
    label_train = torch.from_numpy(label_train0).long()

    train_set = utils.TensorDataset(data_train, label_train)

    return train_set
