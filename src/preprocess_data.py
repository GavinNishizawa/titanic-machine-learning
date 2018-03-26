import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from read_data import read_data


def fill_nulls(df):
    # with age: median, cabin: 'UNKNOWN', Embarked: mode
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Cabin'].fillna('UNKNOWN', inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


'''
Add a column to the dataframe for column: key, with numeric values
'''
def to_numeric(df, key):
    values = list(pd.get_dummies(df[key]).columns)
    df[key+'_n'] = df[key].apply(lambda v: values.index(v))
    return values


def make_numeric(df):
    # convert sex into number
    df['Sex_n'] = df['Sex'].apply( \
        lambda s: int(s.lower().startswith('m')))

    # convert embarked into number and one-hot encoding
    evs = to_numeric(df, 'Embarked')
    df['IsEmb_C'] = df['Embarked_n'].apply( \
        lambda e: 1 if e == evs.index('C') else 0)
    df['IsEmb_Q'] = df['Embarked_n'].apply( \
        lambda e: 1 if e == evs.index('Q') else 0)
    df['IsEmb_S'] = df['Embarked_n'].apply( \
        lambda e: 1 if e == evs.index('S') else 0)


def process_name(df):
    # split name into first and last name
    split_name = df['Name'].str.split(',', expand=True)
    df['Lastname'] = split_name[0]
    df['Name'] = split_name[1]

    # split first name into title and first names
    split_first = df['Name'].str.split('.', expand=True)
    df['Title'] = split_first[0].str.strip()
    df['Name'] = split_first[1]

    # keep only titles occuring more than 10 times (top 4)
    top_titles = list(dict(df['Title'].value_counts()[0:4]).keys())

    # convert title to a numeric value
    df['Title_n'] = df['Title'].apply(lambda t: \
        top_titles.index(t) if t in top_titles else None)

    # for non top title use Sex_n to assign title
    df.loc[df['Title_n'].isnull(), 'Title_n'] = \
        df[df['Title_n'].isnull()]['Sex_n'].apply(lambda s:
            top_titles.index('Mr') if s else top_titles.index('Miss'))

    # convert title number to one-hot encoding
    df['IsMr'] = df['Title_n'].apply(lambda t: \
        1 if t == top_titles.index('Mr') else 0)
    df['IsMiss'] = df['Title_n'].apply(lambda t: \
        1 if t == top_titles.index('Miss') else 0)
    df['IsMrs'] = df['Title_n'].apply(lambda t: \
        1 if t == top_titles.index('Mrs') else 0)
    df['IsMaster'] = df['Title_n'].apply(lambda t: \
        1 if t == top_titles.index('Master') else 0)

    # remove (, ), and " characters
    df['Name'] = df['Name'].replace( \
        to_replace=['\(','\)','\"'], value='',
        regex=True).str.strip()

    # calculate popularity of first letter of name
    df['FirstLetter'] = df['Name'].apply(lambda n: n[0])
    fl_counts = dict(df['FirstLetter'].value_counts())
    df['FL_VC'] = df['FirstLetter'].apply(lambda fl: fl_counts[fl])

    # number of names
    df['Num_names'] = df['Name'].str.split(' ', expand=True).count(axis=1)

    # convert first name to number
    df['Firstname'] = df['Name'].str.split(' ', expand=True)[0]
    names = to_numeric(df, 'Firstname')


def process_cabin(df):
    df['Num_cabins'] = df['Cabin'].str.split(' ', expand=True).count(axis=1)
    df['CabinLetter'] = df['Cabin'].str[0]
    cls = to_numeric(df, 'CabinLetter')


def process_data(df):
    # process the data and extract additional features
    fill_nulls(df)
    make_numeric(df)
    process_name(df)
    process_cabin(df)


def main():
    td = read_data('train.csv')

    process_data(td)

    # display correlations with Survived for numeric data
    s_corr = pd.DataFrame(td.corr()['Survived'])
    print(s_corr)

    # plot correlation heatmap with seaborn
    sns.heatmap(s_corr, annot=True)
    fn = os.path.join('out','survived_corr.png')
    plt.savefig(fn, bbox_inches='tight')
    plt.close()

    # plot absolute correlation heatmap with seaborn
    sns.heatmap(s_corr.abs(), annot=True)
    fn = os.path.join('out','survived_corr_abs.png')
    plt.savefig(fn, bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    main()
