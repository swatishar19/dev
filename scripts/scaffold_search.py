from rdkit import Chem
import pandas as pd

## Cluster 0
scaff1 = Chem.MolFromSmarts('Fc1ccccc1')
scaff2 = Chem.MolFromSmarts('Clc1ccccc1')
scaff3 = Chem.MolFromSmarts('Oc1ccc(C(F)(F)F)cc1') # only 2 chemicals
scaff4 = Chem.MolFromSmarts('[H][n]1cnc([H][n])c1')  # work on this
scaff5 = Chem.MolFromSmarts('[NH]c1ncccc1')  # has only 3 compounds from the training set
scaff6 = Chem.MolFromSmarts('C=Nc1ccccc1')  # only 4 chemicals
scaff7 = Chem.MolFromSmarts('c1ncccn1')
scaff07 = Chem.MolFromSmarts('S(=O)(=O)c1ccccc1')   #('[OH]c1ccccc1')


# print(m.HasSubstructMatch(scaff1) and m.HasSubstructMatch(scaff2))

## Cluster 9
scaff8 = Chem.MolFromSmarts('Clc1ccccc1')
scaff9 = Chem.MolFromSmarts('c1ccc(C(=N))cc1')
scaff10 = Chem.MolFromSmarts('c1ccc(C(=O))cc1')
scaff11 = Chem.MolFromSmarts('C(Cl)(Cl)')
#### the below scaffold cannot be found ####
scaff90 = Chem.MolFromSmarts('C(F)(F)(F)c1cc(N(=O)=O)cc(N(=O)=O)c1')#'c1c(N(=O)=O)cc(C(F)(F)F)cc(N(=O)=O)1')#'N(=O)(=O)c1cc(N(=O)(=O))cc(C(F)(F)F)c1')
scaff91 = Chem.MolFromSmarts('c1ccc(N=C)cc1') #just 3 chemicals in the training set
scaff92 = Chem.MolFromSmarts('c1ccc(NC=O)cc1')
scaff93 = Chem.MolFromSmarts('n1cncn1')

## Cluster 10
scaff12 = Chem.MolFromSmarts('Oc1ccc(C(=O))cc1')
# scaff13 = Chem.MolFromSmarts('Oc1ccc(C(=C))cc1')
scaff13 = Chem.MolFromSmarts('Cc1ccc(Cl)cc(Cl)1')
scaff100 = Chem.MolFromSmarts('C(Cl)(Cl)(Cl)')
scaff101 = Chem.MolFromSmarts('Nc1ccccc1')


## Cluster 11
scaff14 = Chem.MolFromSmarts('Nc1nc(N)ccn1')

## Cluster 1
scaff15 = Chem.MolFromSmarts('S(=O)=O')
scaff16 = Chem.MolFromSmarts('P=O')
scaff117 = Chem.MolFromSmarts('N(=O)(=O)c1ccccc1')

## cluster 2
scaff17 = Chem.MolFromSmarts('SN(C=O)')
scaff18 = Chem.MolFromSmarts('N(C=O)SC(Cl)')

## cluster 4
scaff19 = Chem.MolFromSmarts('c1ccc(CC(N)=O)cc1')  # not present in test set
scaff20 = Chem.MolFromSmarts('P(=O)O')
scaff21 = Chem.MolFromSmarts('O=CNC(=O)NC') # not present in test set
scaff401 = Chem.MolFromSmarts('S(=O)(=O)')

## cluster 6
scaff22 = Chem.MolFromSmarts('[OH]c1ccccc1') #13/77 training set and 10 in test set
scaff222 = Chem.MolFromSmarts('Sc1ccccc1')

## cluster 7
scaff23 = Chem.MolFromSmarts('Nc1ccccc1')
scaff24 = Chem.MolFromSmarts('Clc1ccccc1')   # a lot of compounds have both 23 & 24
scaff71 = Chem.MolFromSmarts('C=Nc1ccccc1')
scaff72 = Chem.MolFromSmarts('[HN]C(=S)N')
scaff73 = Chem.MolFromSmarts('Nc1ccccc1')  # 43 chemicals have this
scaff74 = Chem.MolFromSmarts('[OH]c1ccccc1')
scaff75 = Chem.MolFromSmarts('n1ncnc1')
scaff76 = Chem.MolFromSmarts('c1ccc([NH]c2ccccc2)cc1')

## cluster 8
scaff25 = Chem.MolFromSmarts('[OH]c1ccccc1') # 16/55 have this in the training set
scaff80 = Chem.MolFromSmarts('c1ccc(CNc2ccccc2)cc1')
scaff81 = Chem.MolFromSmarts('[NH]C=O')

## Cluster 15
scaff26 = Chem.MolFromSmarts('c1c2Sc3ccccc3Nc2ccc1')
scaff150 = Chem.MolFromSmarts('Oc1ccccc1')

## Cluster 16
scaff27 = Chem.MolFromSmarts('[OH]c1ccccc1')
scaff160 = Chem.MolFromSmarts('c1ccc(C(=O))cc1')
scaff161 = Chem.MolFromSmarts('c1ccc(O)cc1')

## Cluster 17
scaff28 = Chem.MolFromSmarts('Nc1ccccc1')
scaff170 = Chem.MolFromSmarts('[OH]c1ccccc1')
scaff171 = Chem.MolFromSmarts('c1nccs1')
scaff172 = Chem.MolFromSmarts('n1cncc1')
scaff173 = Chem.MolFromSmarts('n1ccccc1')

## Cluster 19
scaff29 = Chem.MolFromSmarts('C(Cl)(Cl)(Cl)')
scaff30 = Chem.MolFromSmarts('Fc1ccccc1')

## cluster 20
scaff31 = Chem.MolFromSmarts('C(F)(F)(F)c1cc(Cl)cnc1')  # has only 3 chemicals from training
scaff32 = Chem.MolFromSmarts('S(=O)(=O)c1ccccc1')
scaff33 = Chem.MolFromSmarts('Clc1ccccc1')
scaff34 = Chem.MolFromSmarts('c1ccc(O)cc1')
scaff35 = Chem.MolFromSmarts('Nc1ccccc1')

## cluster 21
scaff36 = Chem.MolFromSmarts('S(=O)(=O)')
scaff37 = Chem.MolFromSmarts('N(=O)(=O)c1ccc(O)cc1')

cluster = 'cluster_7'
scaffold = scaff72

training = pd.read_csv(cluster+'to predict structure MultiCase.csv')
test = pd.read_csv(cluster+'testing set for scaffold.csv')
training_profile = pd.read_csv(cluster+'_training_set_profile.csv')
testset_profile = pd.read_csv(cluster+'_testing_set_profile.csv')
# print(test.head())

# chem = []
# for z in range(len(training)):
#     m = Chem.MolFromSmiles(training.iloc[z, -1])
#     if m.HasSubstructMatch(scaffold):
#         chem.append(training.iloc[z,1])
#     # if training.iloc[z, 1] == 'Dextox_601':
#     #     print(training.iloc[z,-1])
# print(chem)

# print(Chem.MolFromSmiles('CCCCCCCCN1SC(Cl)=C(Cl)C1=O').HasSubstructMatch(scaff18))

profile = []
for i in range(len(training)):
    for j in range(len(training_profile)):
        m = Chem.MolFromSmiles(training.iloc[i, -1])
        if m.HasSubstructMatch(scaffold): # and m.HasSubstructMatch(scaff24):

            if training.iloc[i, 1] == training_profile.iloc[j, 0]:
                profile.append(training_profile.iloc[j,:])

for a in range(len(test)):
    for b in range(len(testset_profile)):
        n = Chem.MolFromSmiles(test.iloc[a, -1])
        if n.HasSubstructMatch(scaffold): # and m.HasSubstructMatch(scaff24):

            if test.iloc[a, 1] == testset_profile.iloc[b, 0]:
                profile.append(testset_profile.iloc[b,:])

        # if training.iloc[i,1] =='Dextox_767':
        #     print(training.iloc[i,-1])
profile_df = pd.DataFrame(profile)

# print(profile_df.tail())
modelresults = pd.read_csv(cluster+'to predict structure MultiCase.csv')
modelresults.rename(columns = {'Index':'ID' }, inplace = True)
# print(profile_df.head())
modelpredictions = pd.read_csv(cluster+'testing set for scaffold.csv')
modelpredictions.rename(columns = {'CAS_RN':'ID' }, inplace = True)

# print('NNNNNNNNNNNNNNNNNNNNNNNNNNN')
# print(modelresults.head())
# print('SSSSSSSSSSSSSSSSSSSSSSSSSSS')
# print(modelpredictions.head())
final_df = pd.merge(profile_df, modelresults, on = ['ID'], how = 'left')
# print(final_df.head())
final_df = final_df.drop(['Unnamed: 0','MulticaseActivity', 'chemical_casrn'], axis = 1)
final_df = pd.merge(final_df, modelpredictions, on= ['ID'], how = 'left')
final_df = final_df.drop(['Unnamed: 0','MulticaseActivity'], axis = 1)
# print(final_df.head())
print(final_df.tail())
final_df.to_csv('scaffoldprofile.csv')


# print(Chem.MolFromSmiles('CCCCCCCCN1SC(Cl)=C(Cl)C1=O').HasSubstructMatch(scaff18))