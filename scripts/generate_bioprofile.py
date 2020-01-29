from ciip.ciip.ciip.ciip import *
from ciip.ciip.ciiprofile.ciiprofiler import *
import pandas as pd
from rdkit import Chem
from ciip.ciip.datasets.datasets import Datasets as DS
import os

import os
print(os.getcwd())

data = pd.read_csv('curated dev tox_allspecies shengde,fda,caeser,toxref.txt', sep='\t')

data_ = data[['SMILES','activity']]
print(data_.head())

smiles = data_['SMILES'].tolist()
print(smiles)
pipe = CIIProPreprocess()
print("CIIProPreprocess")
pipe = pipe.addProcess(ProcessCIDS())
print("pipe.addProcess_CIDS")
pipe = pipe.addProcess(ProcessAssays())
print("pipe.addProcess_processAssays")
profile = pd.DataFrame(pipe.run(smiles))
print("pipe.run(smiles)")
profile = profile.fillna(0)

profile.to_csv('devtox_shengdeFDAtoxCAESER_bioprofile_trainingSet.csv')




