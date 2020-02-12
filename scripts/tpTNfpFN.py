import pandas as pd

model_predictions = pd.read_csv('testset_pubchem_missingdata_model_1347031_Nunknown_svc_AID1347031_balanced_training_rdkit_Activity_0_pipeline_prediction_set.csv')
testset = pd.read_csv('testset_pubchem_missingdata_model_1347031_Nunknown.csv')

TP = 0
TN = 0
FP = 0
FN = 0
merged = pd.merge(model_predictions, testset, on = ['Name'], how= 'left')
print(merged.head())

for i in range(len(merged)):
    if int(merged.loc[i, 'DevtoxActivity']) == 1 and int(merged.loc[i, 'ActivityPred']) == 1: #-2 column is predicted toxicity and -1 is true toxicity
        TP +=1
    elif int(merged.loc[i, 'DevtoxActivity']) == 0 and int(merged.loc[i, 'ActivityPred']) == 0:
        TN +=1
    elif int(merged.loc[i, 'DevtoxActivity']) == 0 and int(merged.loc[i, 'ActivityPred']) == 1:
        FP +=1
    elif int(merged.loc[i, 'DevtoxActivity']) == 1 and int(merged.loc[i, 'ActivityPred']) == 0:
        FN +=1

print('TP:', TP,
      'TN:', TN,
      'FP:', FP,
      'FN:', FN)