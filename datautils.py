import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_PAMAP2(fold=None, axis=0, loc='datasets/PAMAP2'):
    label_embeddings = np.load(loc+'/'+'label_embeddings.npy')
    if fold:
        loc = 'datasets/PAMAP2/' + fold

    train_file = os.path.join(loc, "X_train.npy")
    train_labels_file = os.path.join(loc, "y_train.npy")
    test_file = os.path.join(loc, "X_test.npy")
    test_labels_file = os.path.join(loc, "y_test.npy")
    train = np.load(train_file)
    train_labels = np.load(train_labels_file)
    test = np.load(test_file)
    test_labels = np.load(test_labels_file)

    if axis is not None:
        train = train[:, :, axis]
        test = test[:, :, axis]
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std

    train_embeddings = get_embeddings_per_datapoint(train_labels, label_embeddings)
    test_embeddings = get_embeddings_per_datapoint(test_labels, label_embeddings)

    print(train.shape, train_labels.shape, train_embeddings.shape,
          test.shape, test_labels.shape, test_embeddings.shape, label_embeddings.shape)
    return train, train_labels, train_embeddings, test, test_labels, test_embeddings, label_embeddings

def get_embeddings_per_datapoint(y, label_embeddings):
    embeddings = np.zeros((y.shape[0], label_embeddings.shape[1]))
    for i in range(y.shape[0]):
        embeddings[i] = np.copy(label_embeddings[y[i]])
    return embeddings