import pandas as pd
from pubchempy import *
AID = '1259242'

assay = pd.read_csv('AID_' + AID + '_datatable_all.csv', low_memory= False)
assay = assay.drop(assay.index[[0, 1, 2, 3, 4]])
assay = assay[['PUBCHEM_CID', 'PUBCHEM_ACTIVITY_OUTCOME']]
# print(len(assay))
assay = assay.dropna(subset = ['PUBCHEM_CID'])
assay = assay[assay.PUBCHEM_ACTIVITY_OUTCOME != 'Inconclusive']
# print(len(assay))

activity = []

for i in range(len(assay)):
    if assay.iloc[i, 1].lower() == 'active':
        activity.append(1)
    elif assay.iloc[i, 1] == 'Inactive':
        activity.append(0)

assay['Activity'] = activity

print(assay.head(5))


smiles = []
for j in range(len(assay)):
    comp = assay.iloc[j,0]
    c = Compound.from_cid(int(comp))
    print(c.isomeric_smiles)
    smiles.append(c.isomeric_smiles)
assay['SMILES'] = smiles

assay.to_csv('AID_' + AID + '_smiles.csv')