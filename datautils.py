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


def load_PAMAP2(fold=None, axis=None, loc='datasets/PAMAP2'):
    embeddings = np.load(loc+'/'+'label_embeddings.npy')
    if fold:
        loc = 'datasets/PAMAP2/' + fold
    if axis:
        loc += '/' + str(axis)
    train_file = os.path.join(loc, "X_train.npy")
    train_labels_file = os.path.join(loc, "y_train.npy")
    test_file = os.path.join(loc, "X_test.npy")
    test_labels_file = os.path.join(loc, "y_test.npy")
    train = np.load(train_file)
    train_labels = np.load(train_labels_file)
    test = np.load(test_file)
    test_labels = np.load(test_labels_file)


    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std

    print(train.shape, train_labels.shape, test.shape, test_labels.shape, embeddings.shape)
    return train, train_labels, test, test_labels, embeddings
