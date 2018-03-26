import numpy as np
import random as r
from read_data import read_data


def split_data(data, ratio):
    shuffled = data.sample(frac=1)
    s_ind = int(len(shuffled)*ratio)

    train = shuffled[:s_ind]
    test = shuffled[s_ind:]

    return train, test


def main():
    td = read_data('train.csv')

    train, test = split_data(td, 0.7)
    print(train.shape, test.shape)


if __name__=='__main__':
    main()

