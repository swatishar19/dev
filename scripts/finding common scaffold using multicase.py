import pandas as pd

''' For the purpose of multicase the TP & TN predictions were considered 0 for activity
and FP & FN were considered as 1 '''
cluster = 'cluster_21'
CVpred = pd.read_csv(cluster+'_complicated_prediction.csv')
CVpred = CVpred.dropna(subset=['y_pred'])

testing_set = pd.read_csv('curated_testset_FDAdev_removedOverlappingWithTraining.csv', encoding='ISO-8859-1', low_memory= False)
# print(CVpred.columns)

TP = []

for i in range(len(CVpred)):
    if int(CVpred.iloc[i, -2]) == 1 and int(CVpred.iloc[i, -1]) == 1: #-2 column is predicted toxicity and -1 is true toxicity
        TP.append([CVpred.iloc[i, -3], 1, 'TP'])
    elif int(CVpred.iloc[i, -2]) == 0 and int(CVpred.iloc[i, -1]) == 0:
        TP.append([CVpred.iloc[i, -3], 1, 'TN'])
    elif int(CVpred.iloc[i, -2]) == 1 and int(CVpred.iloc[i, -1]) == 0:
        TP.append([CVpred.iloc[i, -3], 0, 'FP'])
    elif int(CVpred.iloc[i, -2]) == 0 and int(CVpred.iloc[i, -1]) == 1:
        TP.append([CVpred.iloc[i, -3], 0, 'FN'])

truepos = pd.DataFrame(TP, columns = ['CAS_RN', 'MulticaseActivity', 'TP/FP/TN/FN'])
predicted_compds = pd.merge(truepos, testing_set, on='CAS_RN')
predicted_compds = predicted_compds[['CAS_RN', 'MulticaseActivity', 'TP/FP/TN/FN',  'SMILES']]
print(len(predicted_compds))

predicted_compds.to_csv(cluster+'testing set for scaffold.csv')
# print(TP)