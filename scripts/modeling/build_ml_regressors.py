from config import directory_check
from molecules_and_features import make_dataset
from classic_ml import split_train_test, REGRESSOR_ALGS, CLASSIFIER_ALGS
from deep_learning import get_regress_stats, regress_scoring, class_scoring

from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, r2_score, make_scorer
from sklearn.externals import joblib

import pandas as pd
import os
import argparse
import numpy as np
import math

parser = argparse.ArgumentParser(description='Build QSAR Models')
parser.add_argument('-ds', '--dataset', metavar='ds', type=str,
                    help='training set name')
parser.add_argument('-f', '--features', metavar='f', type=str,
                    help='features to build model with')
parser.add_argument('-ns', '--n_splits', metavar='ns', type=int,
                    help='number of splits for cross validation')
parser.add_argument('-dd', '--data_dir', metavar='dd', type=str,
                    help='envronmental variable of project directory')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str,
                    help='name of name column in sdf file')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str,
                    help='end point to model')
parser.add_argument('-ts', '--test_set_size', metavar='ts', type=float,
                    help='size of the test set')

args = parser.parse_args()

dataset = args.dataset
features = args.features
n_splits = args.n_splits
seed = 0
env_var = args.data_dir
data_dir = os.getenv(env_var)
name_col = args.name_col
endpoint = args.endpoint
test_set_size = args.test_set_size

# Check to see if necessary directories are present and if not, create them
directory_check(data_dir)

# get training data and split in training, test
# and use a seed for reproducibility
X, y_regress, y_class = make_dataset('{}.sdf'.format(dataset),
                                     data_dir=env_var,
                                     features=features,
                                     name_col=name_col,
                                     endpoint=endpoint,
                                     regress=True,
                                     threshold=2000)

X_train, y_train_regress, X_test, y_test_regress = split_train_test(X, y_class, n_splits, test_set_size, seed, None)

if test_set_size != 0:
    y_train_regress = y_regress.loc[X_train.index].values.ravel()
    y_test_regress = y_regress.loc[X_test.index].values.ravel()

y_train_regress = y_regress.loc[X_train.index]

cv = model_selection.KFold(shuffle=True, n_splits=n_splits, random_state=seed)

for name, clf, params in REGRESSOR_ALGS:
    pipe = pipeline.Pipeline([
        ('scaler', StandardScaler()),
        (name, clf)])
    grid_search = model_selection.GridSearchCV(pipe,
                                               param_grid=params,
                                               cv=cv,
                                               scoring=regress_scoring,
                                               refit='r2_score')
    grid_search.fit(X_train, y_train_regress)

    best_estimator = grid_search.best_estimator_

    print("=======Results for {}=======".format(name))

    # get the predictions from the best performing model in 5 fold cv
    # TODO get predicted probas
    cv_predictions = cross_val_predict(best_estimator, X_train, y_train_regress, cv=cv)
    five_fold_stats = get_regress_stats(None, y_train_regress, cv_predictions)

    # record the predictions and the results
    pd.Series(cv_predictions, index=y_train_regress.index).to_csv(os.path.join(data_dir,
                                                                             'predictions',
                                                                             '{}_{}_{}_{}fcv_predictions.csv'.format(name, dataset,
                                                                                                             features,
                                                                                                             n_splits)))

    # print the 5-fold cv accuracy and manually calculated accuracy to ensure they're correct
    print("Best 5-fold cv r2_score: {}".format(grid_search.best_score_))

    print("All 5-fold results:")
    for score, val in five_fold_stats.items():
        print(score, val)

    # write 5-fold cv results to csv
    pd.Series(five_fold_stats).to_csv(
        os.path.join(data_dir, 'results', '{}_{}_{}_{}fcv_results.csv'.format(name, dataset, features, n_splits)))

    # make predictions on training data, then test data
    train_preds = best_estimator.predict(X_train.values)
    if test_set_size != 0:
        test_preds = best_estimator.predict(X_test.values)
    else:
        test_preds = None

    # write it all to for later
    pd.Series(train_preds, index=y_train_regress.index).to_csv(
        os.path.join(data_dir, 'predictions', '{}_{}_{}_train_predictions.csv'.format(name,
                                                                                          dataset,
                                                                                          features)))
    if test_preds is not None:
        pd.Series(test_preds, index=y_regress.loc[X_test.index].index).to_csv(
            os.path.join(data_dir, 'predictions', '{}_{}_{}_test_predictions.csv'.format(name,
                                                                                         dataset,
                                                                                         features)))

    print("Training data prediction results: ")
    train_stats = get_regress_stats(best_estimator, X_train, y_train_regress)
    for score, val in train_stats.items():
        print(score, val)    # write training predictions and stats also

    if test_preds is not None:
        print("Test data results")
        test_stats = get_regress_stats(best_estimator, X_test, y_test_regress)
        for score, val in test_stats.items():
            print(score, val)
    else:
        test_stats ={}
        for score, val in train_stats.items():
            test_stats[score] = np.nan

    pd.DataFrame([train_stats, test_stats], index=['Training', 'Test']).to_csv(
        os.path.join(data_dir, 'results', '{}_{}_{}_{}_train_test_results.csv'.format(name, dataset, features,
                                                                                      endpoint)))

    # save model
    save_dir = os.path.join(data_dir, 'ML_models', '{}_{}_{}_{}_pipeline.pkl'.format(name, dataset, features, endpoint))
    joblib.dump(best_estimator, save_dir)

