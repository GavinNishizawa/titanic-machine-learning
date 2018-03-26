import numpy as np
import os
from read_data import read_data
from split_data import split_data
from preprocess_data import process_data, get_best_s_corr
from sklearn import ensemble


def main():
    # read in training data and create a random split
    train = read_data('train.csv')
    test = read_data('test.csv')

    # preprocess data
    process_data(train)
    process_data(test)

    # use best correlations for predictions
    b_corr = get_best_s_corr(train)
    b_corr.remove('Survived')

    train_x = train[b_corr]
    train_y = train['Survived']
    test_x = test[b_corr]

    # predict with Random Forest
    rfc = ensemble.RandomForestClassifier()
    rfc.fit(train_x, train_y)
    test['Survived'] = rfc.predict(test_x)

    pred_fn = os.path.join('out','predictions.csv')
    test[['PassengerId','Survived']].to_csv(pred_fn, index=False)


if __name__=='__main__':
    main()

