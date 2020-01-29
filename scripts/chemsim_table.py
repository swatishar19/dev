import pandas as pd
import numpy as np

chemsim = pd.read_csv('chemical similarity training set.csv')

compdC = []
compdD = []
chem_sim_ls = []

list_colnames = list(chemsim)
# print(list(chemsim))
for i in range(len(chemsim)):
    for j in range(i+2, len(chemsim)):
# i+2 because the 0th column is compd names, [0,1] will match 0th element with itself
        if chemsim.iloc[i,j] >= 0.75:
            compdC.append(chemsim.iloc[i,0])
            compdD.append(list_colnames[j])
            chem_sim_ls.append(chemsim.iloc[i,j])

# print(len(compdC), len(compdD), len(chem_sim_ls))

dictionary = {'compound C': compdC , 'compound D': compdD, 'chemical_similarity': chem_sim_ls}
chem_df = pd.DataFrame(dictionary)
# chem_df.to_csv('chem nearest neighbors.csv')


curated_dataset = pd.read_csv('curated_devtox_allspecies_Shengde,tox,fda,caeser.csv')

compdC_act = []
compdD_act = []

# print(list(chem_df))
#
# print(list(curated_dataset))


for i in range(0, len(chem_df)):
    for j in range(len(curated_dataset)):
        if chem_df.iloc[i, -2] == curated_dataset.iloc[j, -1]:
            compdC_act.append(curated_dataset.iloc[j, 1])

        if chem_df.iloc[i, -1] == curated_dataset.iloc[j, -1]:
            compdD_act.append(curated_dataset.iloc[j, 1])

# print(len(compdB_act))
# print(bio_nn.iloc[0,:])
# print(list(curated_dataset))

all = pd.DataFrame(chem_df.iloc[:, :])
# print(all.head())

all['compound C activity'] = compdC_act
all['compound D activity'] = compdD_act

# all.to_csv('chem nn with activity_allCompd.csv')


#############################################################################
### finding the activity cliffs in the chemical similar nearest neighbors ###

chem_actcliffs = all[all['compound C activity'] != all['compound D activity']]

print(chem_actcliffs.head())