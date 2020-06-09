from config import directory_check
from molecules_and_features import make_dataset
from classic_ml import split_train_test, CLASSIFIER_ALGS
from deep_learning import get_class_stats, class_scoring

from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.externals import joblib

import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Build QSAR Models')
parser.add_argument('-ds', '--dataset', metavar='ds', type=str,
                    help='training set name')
parser.add_argument('-f', '--features', metavar='f', type=str,
                    help='features to build model with')
parser.add_argument('-ns', '--n_splits', metavar='ns', type=int,
                    help='number of splits for cross validation')
parser.add_argument('-dd', '--data_dir', metavar='dd', type=str,
                    help='environmental variable of project directory')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str,
                    help='name of name column in sdf file')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str,
                    help='end point to model')
parser.add_argument('-t', '--threshold', metavar='t', type=int,
                    help='threshold cutoff')
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
threshold = args.threshold
test_set_size = args.test_set_size

# Check to see if necessary directories are present and if not, create them
directory_check(data_dir)

# get training data and split in training, test
# and use a seed for reproducibility
X, y = make_dataset('{}.sdf'.format(dataset),
                    data_dir=env_var,
                    features=features,
                    name_col=name_col,
                    endpoint=endpoint,
                    threshold=threshold)
X_train, y_train_class, X_test, y_test_class = split_train_test(X, y, n_splits, test_set_size, seed, None)

cv = model_selection.StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=seed)

print("Num Actives: {}".format((y==1).sum()))
print("Num Inactives: {}".format((y==0).sum()))

for name, clf, params in CLASSIFIER_ALGS:
    pipe = pipeline.Pipeline([
        ('scaler', StandardScaler()),
        (name, clf)])
    grid_search = model_selection.GridSearchCV(pipe,
                                               param_grid=params,
                                               cv=cv,
                                               scoring=class_scoring,
                                               refit='ACC')
    grid_search.fit(X_train, y_train_class)
    best_estimator = grid_search.best_estimator_

    print("=======Results for {}=======".format(name))

    # get the predictions from the best performing model in 5 fold cv
    # TODO get predicted probas
    cv_predictions = cross_val_predict(best_estimator, X_train, y_train_class, cv=cv)
    five_fold_stats = get_class_stats(None, y_train_class, cv_predictions)

    # record the predictions and the results
    pd.Series(cv_predictions, index=y_train_class.index).to_csv(os.path.join(data_dir,
                                                                             'predictions',
                                                                             '{}_{}_{}_{}_{}_{}fcv_predictions.csv'.format(name, dataset,
                                                                                                             features,
                                                                                                             endpoint,
                                                                                                             threshold,
                                                                                                             n_splits)))

    # print the 5-fold cv accuracy and manually calculated accuracy to ensure they're correct
    print("Best 5-fold cv ACC: {}".format(grid_search.best_score_))

    print("All 5-fold results:")
    for score, val in five_fold_stats.items():
        print(score, val)

    # write 5-fold cv results to csv
    pd.Series(five_fold_stats).to_csv(
        os.path.join(data_dir, 'results', '{}_{}_{}_{}_{}_{}fcv_results.csv'.format(name,
                                                                                  dataset,
                                                                                  features,
                                                                                  endpoint,
                                                                                  threshold,
                                                                                  n_splits)))


    # make predictions on training data, then test data
    train_preds = best_estimator.predict(X_train.values)
    if test_set_size != 0:
        test_preds = best_estimator.predict(X_test.values)
    else:
        test_preds = None

    # write it all to for later
    pd.Series(train_preds, index=y_train_class.index).to_csv(
        os.path.join(data_dir, 'predictions', '{}_{}_{}_{}_{}_train_predictions.csv'.format(name, dataset, features,
                                                                                            endpoint, threshold)))
    if test_preds is not None:
        pd.Series(test_preds, index=y_test_class.index).to_csv(
            os.path.join(data_dir, 'predictions', '{}_{}_{}_{}_{}_test_predictions.csv'.format(name, dataset, features,
                                                                                               endpoint, threshold)))

    print("Training data prediction results: ")
    train_stats = get_class_stats(best_estimator, X_train, y_train_class)
    for score, val in train_stats.items():
        print(score, val)    # write training predictions and stats also

    if test_preds is not None:
        print("Test data results")
        test_stats = get_class_stats(best_estimator, X_test, y_test_class)
        for score, val in test_stats.items():
            print(score, val)
    else:
        test_stats ={}
        for score, val in train_stats.items():
            test_stats[score] = np.nan

    pd.DataFrame([train_stats, test_stats], index=['Training', 'Test']).to_csv(
        os.path.join(data_dir, 'results', '{}_{}_{}_{}_{}_train_test_results.csv'.format(name, dataset, features,
                                                                                         endpoint, threshold)))

    # save model
    save_dir = os.path.join(data_dir, 'ML_models', '{}_{}_{}_{}_{}_pipeline.pkl'.format(name, dataset, features,
                                                                                        endpoint, threshold))
    joblib.dump(best_estimator, save_dir)

