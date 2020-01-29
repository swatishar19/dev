import os
import argparse
import pandas as pd

from deep_learning import get_class_stats
from molecules_and_features import make_dataset

from sklearn.externals.joblib import load

parser = argparse.ArgumentParser(description='Use QSAR Models to Predict Data')

parser.add_argument('-dd', '--data_dir', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ps', '--prediction_set', metavar='ps', type=str, help='prediction set name')
parser.add_argument('-ev', '--evaluate', metavar='ev', type=str, help='calculate statistics using external set? (Y/N)')
parser.add_argument('-ep', '--endpoints', metavar='ep', type=str, help='endpoints to model')
parser.add_argument('-t', '--threshold', metavar='t', type=int, help='toxic LD50 threshold (mg/kg)')
parser.add_argument('-ds', '--train_name', metavar='ds', type=str, help='training set name')

args = parser.parse_args()

data_dir = os.getenv(args.data_dir)
env_var = args.data_dir
features = args.features.split(',')
name_col = args.name_col
prediction_set = args.prediction_set
endpoints = args.endpoints.split(',')
threshold = args.threshold
evaluate = args.evaluate
train_name = args.train_name

for alg in ['ada', 'bnb', 'knn', 'nb', 'rf', 'svc']:
    for feature in features:
        for endpoint in endpoints:
            model_name = f'{alg}_{train_name}_{feature}_{endpoint}_{threshold}_pipeline'
            model_file_path = os.path.join(data_dir, 'ML_models', f'{model_name}.pkl')

            if os.path.exists(model_file_path):
                if evaluate in ['Y', 'y']:
                    X_pred_set, y = make_dataset(f'{prediction_set}.sdf', data_dir=env_var, features=feature,
                                                 name_col=name_col, endpoint=endpoint, threshold=threshold)

                else:
                    X_pred_set = make_dataset(f'{prediction_set}.sdf', data_dir=env_var, features=features,
                                              name_col=name_col, endpoint=None, threshold=None, pred_set=True)

                loaded_model = load(model_file_path)
                predictions = loaded_model.predict(X_pred_set)
                probabilities = loaded_model.predict_proba(X_pred_set)
                pd.Series(predictions, index=X_pred_set.index).to_csv(
                    os.path.join(data_dir, 'predictions', f'{prediction_set}_{model_name}_prediction_set.csv'))
                pd.DataFrame(probabilities, index=X_pred_set.index).to_csv(
                    os.path.join(data_dir, 'predictions', f'{prediction_set}_{model_name}_probas_prediction_set.csv'))

                if evaluate in ['Y', 'y']:
                    ext_set_stats = get_class_stats(None, y, predictions)
                    pd.Series(ext_set_stats).to_csv(
                        os.path.join(data_dir, 'results', '{}_{}_results.csv'.format(prediction_set,
                                                                                     model_name)))
                    print("=======Results from External Set Comparison=======")

                    for score, val in ext_set_stats.items():
                        print(score, val)
