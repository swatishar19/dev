from rdkit import Chem
import pandas as pd

dev_data = pd.read_csv('curated_devtox_allspecies_Shengde,tox,fda,caeser.csv')
# print(len(dev_data['SMILES']))

updated_smiles = []

# print(dev_data.iloc[1365,2])
# print(dev_data.iloc[745, :])
# for i in range(745, 1366):
#     updated_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(dev_data.iloc[i, 2])))
# print(len(updated_smiles))



mols = [Chem.MolFromSmiles(smi) for smi in dev_data.SMILES if Chem.MolFromSmiles(smi)]
# dev_data = dev_data.set_index('Dev_Tox_ID')
# print(dev_data.head())
# print("Rdkit created {} mols".format(len(mols)))



from rdkit.Chem import MACCSkeys

fps = pd.DataFrame([list(MACCSkeys.GenMACCSKeys(mol)) for mol in mols])
fps = fps.set_index(dev_data['Dev_Tox_ID'])
# print(fps.head())

from sklearn.metrics.pairwise import pairwise_distances

jac_sim = pd.DataFrame(1 - pairwise_distances(fps, metric="jaccard"))
jac_sim = jac_sim.set_index(dev_data['Dev_Tox_ID'])

jac_sim.columns = jac_sim.index
# print(jac_sim.head())
# print(jac_sim.columns[0])
jac_sim.to_csv('chemical similarity training set.csv')
import matplotlib.pyplot as plt

# plt.imshow(jac_sim, cmap='coolwarm')
# plt.show()