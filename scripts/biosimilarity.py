import pandas as pd

bioprofile = pd.read_csv('filtered_training_devtox_shengdeFDAtoxCAESER_bioprofile_compdNames.csv', index_col = 0)
bioprofile = bioprofile.loc[:, (bioprofile == 1).sum() >= 2]

# def convert(response):
#     if response == 'Active':
#         return 1
#     elif response == 'Inactive':
#         return -1
#     else: return 0
#
# bioprofile = bioprofile.applymap(convert)

def biosimilarity_distances(X, X_, weights=None):
    """ calculate the distance of every element in X in X_ """
    import numpy as np
    X_new = np.concatenate([X, X_])

    active_weights = np.array([1] * X_new.shape[1])
    inactive_weights = active_weights

    if weights == 'actives':
        w = (X_new == -1).sum().sum() / (X_new == 1).sum().sum()
        active_weights = np.array([w] * X_new.shape[1])
    elif weights == 'inactives':
        w = (X_new == 1).sum().sum() / (X_new == -1).sum().sum()
        inactive_weights = np.array([w]  * X_new.shape[1])

    positives = X_new.copy()
    positives[X_new == -1] = 0

    negatives = X_new.copy()
    negatives[X_new == 1] = 0

    weighted_positives = np.multiply(positives, active_weights)
    weighted_negatives = np.multiply(negatives, inactive_weights)

    tot_pos = positives.dot(weighted_positives.T)
    tot_neg = negatives.dot(weighted_negatives.T)
    diff_0 = positives.dot(negatives.T)
    diff = np.minimum(diff_0, diff_0.T)

    numer = tot_pos + tot_neg
    denom = tot_pos + tot_neg - diff

    biosim, conf = np.nan_to_num(1 - (numer / denom)), denom
    return biosim[:len(X), len(X):], conf[:len(X), len(X):]

biosim, conf = biosimilarity_distances(bioprofile.values, bioprofile.values, weights='inactives')

biosim = pd.DataFrame(1 - biosim, index=bioprofile.index, columns=bioprofile.index)
conf = pd.DataFrame(conf, index=bioprofile.index, columns=bioprofile.index)

# print(biosim.head())
biosim.to_csv('biosim_mat_dev_training.csv')
conf.to_csv('confidence_biosim_dev_training.csv')

import matplotlib.pyplot as plt

plt.imshow(biosim)

# plt.show()