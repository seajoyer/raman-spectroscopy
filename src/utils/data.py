import numpy as np


def read_data(file_path):

    return np.loadtxt(file_path, skiprows=1, unpack=True)