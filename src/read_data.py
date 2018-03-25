import numpy as np
import pandas as pd
import os


def main():
  td = pd.read_csv(os.path.join('input', 'train.csv'))
  print(td.head())
  print("="*80)

  # View fields with null values
  print(td.isnull().sum())

  # Fill nulls
  # with age: median, cabin: 'UNKNOWN', Embarked: mode
  td['Age'].fillna(td['Age'].median(), inplace=True)
  td['Cabin'].fillna('UNKNOWN', inplace=True)
  td['Embarked'].fillna(td['Embarked'].mode()[0], inplace=True)
  
  print(td.head())
  

if __name__=='__main__':
  main()
