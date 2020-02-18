from pubchempy import *
import pandas as pd
from rdkit import Chem

### Choose the AID number and the scaffold ###
AID = '1347031'

df = pd.read_csv('AID'+AID+'_balanced_training.csv', encoding='ISO-8859-1', low_memory= False)
scaff = Chem.MolFromSmarts('[*]-[#6](=[*])-[#7](-[*])-[#6](@-[*])-@[*]')

def scaffold_presence(file, scaffold):
    active_scaffold = []
    for i in range(len(file)):
        # file = file[file.Activity == 1]
        m = Chem.MolFromSmiles(file.loc[i, 'SMILES'])
        if m.HasSubstructMatch(scaffold) and file.loc[i, 'Activity'] == 1:
            active_scaffold.append(file.iloc[i, :])
    print(len(active_scaffold))
    return pd.DataFrame(active_scaffold)

df1 = scaffold_presence(df, scaff)
print(df1.head())


def convert_CID_name(df):
    name = []
    for i in range(len(df)):
        # print(i)
        c = Compound.from_cid(str(df.iloc[i, -1]))
        print(c.synonyms[0])
        name.append(c.synonyms[0])


    df['Name'] = name
    print(len(name), len(df))

    df.to_csv(AID+'Nunknown_scaffold_activesonly_trainingSet.csv')

convert_CID_name(df1)






