import numpy as np
import pandas as pd
import os


def main():
  td = pd.read_csv(os.path.join('input', 'train.csv'))
  print(td.head())
  

if __name__=='__main__':
  main()
