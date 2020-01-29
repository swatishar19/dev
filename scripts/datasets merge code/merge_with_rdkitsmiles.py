from rdkit import Chem
import pandas as pd

#print(Chem.MolToSmiles(Chem.MolFromSmiles('CCC(=C1C(=O)CC(CC1=O)C(=O)[O-])[O-].[Ca+2]')))

def rdkit_smiles(filename):
    df = pd.read_csv(filename + '.csv', encoding='ISO-8859-1', low_memory = False)
    #print(df.head())
    updated_smiles = []

    #print(df.iloc[1,10])

    for i in range(0, len(df)):
        updated_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(df.loc[i, 'SMILES'])))
    #print(updated_smiles)

    df['rdkit SMILES'] = updated_smiles
    df.to_csv(filename + '.csv')

#filename = 'ToxRefDB_endpointMatrix_MGRratDEVREP_SMILES_added'
#rdkit_smiles(filename)

def merge_datasets(list_all):
    df_merged = pd.merge(list_all[0], list_all[1], on=['rdkit SMILES', 'chemical_name', 'chemical_casrn'], how='outer')

    #for i in range(2, len(list_all)):
        #df_merged = pd.merge(df_merged, list_all[i], on=['rdkit SMILES', 'NAME', 'CAS_RN'], how='outer')
    df_merged.to_csv('combined_toxrefDB_rep_rats_LEL_LOAEL.csv')

df1 = pd.read_csv('ToxRefDB_endpointMatrix_MGRratDEVREP_SMILES_added.csv', encoding='ISO-8859-1', low_memory = False)
df2 = pd.read_csv('toxref_rep_minloael_rat_mouse_SMILES_added.csv', encoding='ISO-8859-1', low_memory = False)

list_all_datasets = [df1, df2]

merge_datasets(list_all_datasets)
