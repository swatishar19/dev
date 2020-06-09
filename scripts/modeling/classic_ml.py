# the basics
import os
import pandas as pd
import numpy as np

# project imports
from molecules_and_features import make_dataset
from deep_learning import get_class_stats

# basic sklearn stuff
from sklearn import pipeline
from sklearn import model_selection

# preprocessing/data selection
from sklearn.preprocessing import StandardScaler

# unsupervised ml
from sklearn.decomposition import PCA

#supervised ml
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

# plotting
import matplotlib.pyplot as plt

DATA_DIR = os.getenv('NICEATM_ACUTE_ORAL_DATA')

def perform_pca(X, y, dimensions=2, filename=None):
    """ Performs principal deoomposision on matrix X and save the figure as a png.
    Colors data points according to y where 1 == red, 0 == green.
    
    :param X: Pandas dataframe 
    :param y: Pandas Series or Dataframe
    :param filename: filename to save figure, if None will display 
    :param dimensions: Number of dimensions to decompose to
    :return: 
    """
    pipe = pipeline.Pipeline([('scaler', StandardScaler()),
                              ('pca', PCA(dimensions) )])

    decomposed_matrix = pipe.fit_transform(X)

    actives = plt.scatter(decomposed_matrix[(y == 1), 0],
                          decomposed_matrix[(y == 1), 1],
                          color=(1, 0, 0, 0.4))
    inactives = plt.scatter(decomposed_matrix[(y == 0), 0],
                          decomposed_matrix[(y == 0), 1],
                          color=(0, 1, 0, 0.4))

    if filename:
        plt.savefig(os.path.join(DATA_DIR, 'figures', 'pca', filename + '.png'))
    else:
        plt.show()

def split_train_test(X, y, n_split, test_set_size, seed, major_subsample=None):
    """ Splits data into training and test sets

    :param n_split:
    :param test_set_size:
    :param seed:
    :return:
    """
    assert X.shape[0] == y.shape[0], 'The lengths of X and y do not match X == {}, y == {}.'.format(X.shape[0],
                                                                                                    y.shape[0])

    if major_subsample != None:
        if sum(y) > y.shape[0] / 2:
            major_class = 1
            minor_class = 0
        else:
            major_class = 0
            minor_class = 1

        major_class_index = y[y == major_class].index
        major_class_index_remove = np.random.choice(np.array(major_class_index),
                                                    int(major_class_index.shape[0] * (1. - major_subsample)),
                                                    replace=False)
        X.drop(X.index[major_class_index_remove], inplace=True)
        y.drop(y.index[major_class_index_remove], inplace=True)


    # split X into training sets and test set the size of test_set_
    if test_set_size != 0:
        batch_size = int(y.shape[0] * (1 - test_set_size) // n_split)  # calculating batch size
        train_size = int(batch_size * n_split)

        X_train_tmp, X_test, y_train_class_tmp, y_test_class = model_selection.train_test_split(X,
                                                                                                y,
                                                                                                train_size=train_size,
                                                                                                stratify=y,
                                                                                                random_state=seed)
    else:
        X_train_tmp = X
        y_train_class_tmp = y
        X_test = None
        y_test_class = None

    cv = model_selection.StratifiedKFold(shuffle=True, n_splits=n_split, random_state=seed)
    valid_idx = []  # indexes for new train dataset
    for (_, valid) in cv.split(X_train_tmp, y_train_class_tmp):
        valid_idx += valid.tolist()

    X_train = X_train_tmp.iloc[valid_idx]
    y_train_class = y_train_class_tmp.iloc[valid_idx]

    return X_train, y_train_class, X_test, y_test_class

# Algs is a list of tuples
# where item 1 is the name
# item 2 is a scikit-learn machine learning classifiers
# and item 3 is the paramaters to grid search through

seed = 0

CLASSIFIER_ALGS = [
    ('rf', RandomForestClassifier(max_depth=10, # max depth 10 to prevent overfitting
                                  class_weight='balanced',
                                  random_state=seed), {'rf__n_estimators':[5, 10, 25]}),
    ('nb', GaussianNB(), {}),
    ('knn', KNeighborsClassifier(metric='euclidean'), {'knn__n_neighbors':[1, 3, 5],
                                                        'knn__weights': ['uniform', 'distance']}),
    ('svc', SVC(probability=True,
                class_weight='balanced',
                random_state=seed), {'svc__kernel': ['rbf'],
                                     'svc__gamma': [1e-2, 1e-3],
                                     'svc__C': [1, 10]}),
    ('bnb', BernoulliNB(alpha=1.0), {}),
    ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.9, random_state=seed), {})
]


REGRESSOR_ALGS = [
    ('rfr', RandomForestRegressor(max_depth=10, # max depth 10 to prevent overfitting
                                  random_state=seed), {'rfr__n_estimators':[5, 10, 25]}),
    ('knnr', KNeighborsRegressor(metric='euclidean'), {'knnr__n_neighbors':[1, 3, 5],
                                                        'knnr__weights': ['uniform', 'distance']}),
    ('svr', SVR(), {'svr__kernel': ['rbf'],
                                     'svr__gamma': [1e-2, 1e-3],
                                     'svr__C': [1, 10]})
]





if __name__ == '__main__':
    dataset = 'trainingset'
    features = 'rdkit'

    X_train, y_train = make_dataset('{}.sdf'.format(dataset), features)

    #perform_pca(X_train, y_train, filename='training_set_pca')

    X_train, y_train_class, X_test, y_test_class = split_train_test(X_train, y_train, 5, 0.2, 0, None)

    for name, clf, params in CLASSIFIER_ALGS:
        pipe = pipeline.Pipeline([
            ('scaler', StandardScaler()),
            (name, clf)])
        grid_search = model_selection.GridSearchCV(pipe, param_grid=params)
        grid_search.fit(X_train, y_train_class)
        best_estimator = grid_search.best_estimator_

        print("=======Results for {}=======".format(name))
        print("Best 5-fold cv: {}".format(grid_search.best_score_))
        training_data_prediction_results = get_class_stats(best_estimator, X_train, y_train_class)
        print("Training data prediction results: ")
        for val, score in training_data_prediction_results.items():
            print(val, score)

        test_data_prediction_results = get_class_stats(best_estimator, X_test, y_test_class)
        print("Test data prediction results: ")
        for val, score in test_data_prediction_results.items():
            print(val, score)