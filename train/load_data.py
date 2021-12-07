from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf

LABEL_NAME = "gesture"
DATA_NAME = "accel_ms2_xyz"

class DataLoader(object):
    """Loads data and prepares for training."""

    def __init__(
        self, train_data_path, valid_data_path, test_data_path, seq_length, num_features
    ):
        self.dim = num_features
        self.seq_length = seq_length
        self.label2id = {
            "sit/stand": 0,
            "walk": 1,
            "liedown": 2,
            "jump": 3,
            "transition/undefined": 4,
            "stairsup": 5,
            "stairsdown": 6,
        }
        self.train_data, self.train_label, self.train_len = self.get_data_file(
            train_data_path, "train"
        )
        self.valid_data, self.valid_label, self.valid_len = self.get_data_file(
            valid_data_path, "valid"
        )
        self.test_data, self.test_label, self.test_len = self.get_data_file(
            test_data_path, "test"
        )

        self.format()

    def get_data_file(self, data_path, data_type):
        """Get train, valid and test data from files."""
        data = []
        label = []

        with open(data_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                dic = json.loads(line)
                data.append(dic[DATA_NAME])
                label.append(dic[LABEL_NAME])

        length = len(label)
        print(data_type + "_data_length:" + str(length))

        return data, label, length

    def pad(self, data, seq_length, dim):
        """Get neighbour padding."""
        noise_level = 20
        padded_data = []

        # Before- Neighbour padding
        tmp_data = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[0]
        tmp_data[(seq_length - min(len(data), seq_length)):] = data[:min(len(data), seq_length)]
        padded_data.append(tmp_data)

        # After- Neighbour padding
        tmp_data = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[-1]
        tmp_data[: min(len(data), seq_length)] = data[:min(len(data), seq_length)]
        padded_data.append(tmp_data)
        return padded_data

    def format_support_func(self, padded_num, length, data, label):
        """Support function for format.(Helps format train, valid and test.)"""

        # Add 2 padding, initialize data and label
        length *= padded_num
        features = np.zeros((length, self.seq_length, self.dim))
        labels = np.zeros(length)

        # Get padding for train, valid and test
        for idx, (data, label) in enumerate(zip(data, label)):
            padded_data = self.pad(data, self.seq_length, self.dim)
            for num in range(padded_num):
                features[padded_num * idx + num] = padded_data[num]
                labels[padded_num * idx + num] = self.label2id[label]

        # Turn into tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (features, labels.astype("int32"))
        )

        return length, dataset

    def format(self):
        """Format data(including padding, etc.) and get the dataset for the model."""
        padded_num = 2
        self.train_len, self.train_data = self.format_support_func(
            padded_num, self.train_len, self.train_data, self.train_label
        )
        self.valid_len, self.valid_data = self.format_support_func(
            padded_num, self.valid_len, self.valid_data, self.valid_label
        )
        self.test_len, self.test_data = self.format_support_func(
            padded_num, self.test_len, self.test_data, self.test_label
        )