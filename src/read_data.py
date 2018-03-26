import numpy as np
import pandas as pd
import os


def read_data(fn):
    td = pd.read_csv(os.path.join('input', fn))
    return td


def main():
    td = read_data('train.csv')
    print(td.head())


if __name__=='__main__':
    main()
