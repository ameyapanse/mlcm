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

def load_PAMAP2(fold, axis):
    class_to_activity_map = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
    0: "other transient activities"
    }
    v_map = np.vectorize(lambda a: class_to_activity_map.get(a, 'no matching activity found'))
    if not fold :
        loc = 'datasets/PAMAP2'
    else :
        loc = 'datasets/PAMAP2/' + fold
    if axis:
        loc += '/' + str(axis)
    train_file = os.path.join(loc,"X_train.npy")
    train_labels_file = os.path.join(loc, "y_train.npy")
    test_file = os.path.join(loc, "X_test.npy")
    test_labels_file = os.path.join(loc, "y_test.npy")
    train = np.load(train_file)
    train_labels = np.load(train_labels_file)
    train_activities = v_map(train_labels)
    test = np.load(test_file)
    test_labels = np.load(test_labels_file)
    test_activities = v_map(test_labels)

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std

    print(train.shape, train_labels.shape, test.shape, test_labels.shape)
    return train, train_labels, train_activities, test, test_labels, test_activities