import numpy as np
import sys

#TRAIN_DATA_FILE = "dataset/train/Inertial Signals/total_acc_x_train.txt"
TRAIN_DATA_FILE = "dataset/train/Inertial Signals/body_gyro_x_train.txt"


def load_training_data():
    """ Returns a matrix of training data.
    shape of result = (n_exp, len)
    """
    data = np.loadtxt(TRAIN_DATA_FILE)
    print(data.shape)
    return data.T


class DataLoader(object):
    def __init__(self, data, batch_size=128, num_steps=1):
        self.batch_size = batch_size
        self.n_data, self.seq_len = data.shape
        self._data = data[:self.batch_size, :]

        self.num_steps = num_steps
        self._data = self._data.reshape((self.batch_size, self.seq_len, 1))
        self._reset_pointer()

    def _reset_pointer(self):
        self.pointer = 0

    def reset(self):
        self._reset_pointer()

    def has_next(self):
        return self.pointer + self.num_steps < self.seq_len - 1

    def next_batch(self):
        batch_xs = self._data[:, self.pointer:self.pointer + self.num_steps, :]
        batch_ys = self._data[:, self.pointer + 1:self.pointer + self.num_steps + 1, :]
        self.pointer = self.pointer + self.num_steps
        return batch_xs, batch_ys

