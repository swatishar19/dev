from rdkit import Chem
import pandas as pd

filename = 'rep_rat_multiCase_combined data_woDupl.csv'
smiles_col = 'SMILES'

df = pd.read_csv(filename, sep=',') #, encoding='ISO-8859-1')


mols = []
for i, data in df.iterrows():
    mol = Chem.MolFromSmiles(data[smiles_col])
    if mol:
        for prop, val in data.iteritems():
            mol.SetProp(prop, str(val))
        mols.append(mol)

w = Chem.SDWriter(filename.split('.')[0] + '.sdf')
for mol in mols:
    w.write(mol)
w.close()
