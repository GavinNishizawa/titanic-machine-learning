import numpy as np
import pandas as pd
from read_data import read_data


def fill_nulls(df):
    # with age: median, cabin: 'UNKNOWN', Embarked: mode
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Cabin'].fillna('UNKNOWN', inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


def make_numeric(df):
    # convert sex into number
    df['Sex_n'] = df['Sex'].apply( \
        lambda s: int(s.lower().startswith('m')))

    # convert embarked into number and one-hot encoding
    evs = list(pd.get_dummies(df['Embarked']).columns)
    df['Embarked_n'] = df['Embarked'].apply( \
        lambda e: evs.index(e))
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
    names = list(pd.get_dummies(df['Firstname']).columns)
    df['Firstname_n'] = df['Firstname'].apply( \
        lambda n: names.index(n))

    #print(df[['Title', 'Firstname_n', 'Num_names', 'FL_VC', 'Lastname', 'IsMr', 'IsMiss', 'IsMrs', 'IsMaster']].head())


def main():
    td = read_data('train.csv')

    # Fill nulls
    fill_nulls(td)
    make_numeric(td)
    process_name(td)

    print(td.head())


if __name__=='__main__':
    main()
