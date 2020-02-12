import pandas as pd
from rdkit import Chem
from pubchempy import *
import random
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

AID = '1259242'
dataframe = pd.read_csv('AID_'+AID+'_smiles.csv')

def rdkit_smiles_conversion(df):
    updated_smiles = []
    print(len(df))
    for i in range(0, len(df)):
        # print(df.loc[i, 'PUBCHEM_CID'])
        updated_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(df.loc[i, 'SMILES'])))

    df['rdkit SMILES'] = updated_smiles
    df.to_csv('AID' + AID + 'rdkitsmiles.csv')
# rdkit_smiles_conversion(dataframe)

### Remove duplicated using CaseUltra ###

file = pd.read_csv(AID + ' curated.txt', sep = '\t')

def randomly_select_activeInactive (df):

    active = df[df.Activity == 1]
    count_active = len(active)
    inactive = df[df.Activity == 0]
    count_inactive = len(inactive)
    if count_active <= count_inactive:
        random_inactive = inactive.sample(count_active)
        new_df = pd.concat([active, random_inactive])
    elif count_inactive <= count_active:
        random_active = active.sample(count_inactive)
        new_df = pd.concat([inactive, random_active])

    print(len(new_df.Name.unique()), len(new_df))
    new_df.to_csv('AID' + AID + '_balanced_training.csv')

randomly_select_activeInactive(file)

# randomfile = pd.read_csv('AID' + AID + '_balanced_training.csv')

def descriptors (df):
    dfrandom = df.copy()
    dfrandom.index = dfrandom.CID
    dfrandom['rdkit'] = [Chem.MolFromSmiles(smi) if Chem.MolFromSmiles(smi) else None
                         for smi in dfrandom.SMILES]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
    X = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in dfrandom.rdkit],
                     columns = list(calc.GetDescriptorNames()),
                     index = dfrandom.index)
    return X

def load_ECFP6(df):
    dfrandom = df.copy()
    dfrandom.index = dfrandom.CID
    dfrandom['rdkit'] = [Chem.MolFromSmiles(smi) if Chem.MolFromSmiles(smi) else None
                    for smi in dfrandom.SMILES]

    data = []
    for mol in dfrandom.rdkit:
        data.append([int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)])
    X = pd.DataFrame(data, columns=list(range(1024)), index=dfrandom.index)
    return X




