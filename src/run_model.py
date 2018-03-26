import numpy as np
from read_data import read_data
from split_data import split_data
from preprocess_data import process_data, get_best_s_corr
from sklearn import linear_model, metrics


def main():
    # read in training data and create a random split
    td = read_data('train.csv')
    train, test = split_data(td, 0.7)

    # preprocess data
    process_data(train)
    process_data(test)

    # use best correlations for predictions
    b_corr = get_best_s_corr(train)
    b_corr.remove('Survived')

    train_x = train[b_corr]
    train_y = train['Survived']
    test_x = test[b_corr]
    test_y = test['Survived']

    # test Logistic Regression model
    lrm = linear_model.LogisticRegressionCV()
    lrm.fit(train_x, train_y)
    prs = lrm.predict(test_x)
    print("Logistic Regression results")
    print(metrics.classification_report(prs, test_y))
    print(metrics.accuracy_score(prs, test_y))


if __name__=='__main__':
    main()

