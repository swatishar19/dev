import numpy as np
import pandas as pd


def assays_filter_based_on_actives(no_of_actives, df):
    """ Takes a dataframe (df) and will remove assays based on number of active datapoint,
    return the df after filtering"""
    del_lis = []
    remain_lis = []
    bool_columns = (sum(df.values ==1) >= no_of_actives)
    for i in range(len(df.columns)):
        if bool_columns[i] == False:
            del_lis.append(i)
        else:
            remain_lis.append(i)
    remain_df = df.drop(df.columns[del_lis], axis=1)
    return remain_df

def remove_correlated_assays(df, t=0.9, min_responses=10):
    """ Takes a dataframe (df) and will remove assays correlated by a
    threshold (t) and must have min no of responses min_responses """
    # find pearsons correlations and shuffle by rows
    # to avoid biasing selection to the lower rows
    # and set diag to -2 so no features correlated
    # with themselevs get removed
    corrs = df.corr(min_periods=min_responses)
    np.fill_diagonal(corrs.values, -2)
    assays_to_remove = set()
    for row, data in corrs.iterrows():
        if row in assays_to_remove:
            continue
        correlated = data[data >= t].index
        assays_to_remove.update(correlated)
    return df[[col for col in df.columns if col not in assays_to_remove]]

df1 = pd.read_csv('devtox_shengdeFDAtoxCAESER_bioprofile_trainingSet.csv', sep=',', encoding='ISO-8859-1', low_memory= False)
df2 = assays_filter_based_on_actives(5, df1)

df3 = remove_correlated_assays(df2)

df2.to_csv('filtered_trainingset_DEV_shengdeFDAtoxCAESER_bioprofile_woRemovingCorrelatedAssays.csv')