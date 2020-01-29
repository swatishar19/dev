import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA


training = pd.read_csv('curated dev_allspecies shengde,fda,caeser,toxref.csv', sep = ',')
# training = training[['Name', 'casrn', 'SMILES', 'activity']]
training.index = training.NAME

''' Adding rdkit descriptors Training set'''
mols = []
names = []
for name, smiles in training.SMILES.iteritems():
    if Chem.MolFromSmiles(smiles):
        mols.append(Chem.MolFromSmiles(smiles))
        names.append(name)
# print(len(mols))

calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])

descriptors_train = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in mols],
                           columns = list(calc.GetDescriptorNames()), index = names)
descriptors_train = descriptors_train.loc[:, ~descriptors_train.isnull().any()]
# print(descriptors_train.head())



''' Standardizing the descriptors '''
from sklearn.preprocessing import scale

descriptors_train_std = pd.DataFrame(scale(descriptors_train), index = descriptors_train.index, columns = descriptors_train.columns)
# print(descriptors_train_std)

''' Adding descriptors for test set'''
test = pd.read_csv('fda_dev_testset_noerrors.csv', sep = ',', encoding='ISO-8859-1', low_memory= False)
test.index = test.Name

mols = []
names = []
for name, smiles in test.SMILES.iteritems():
    if Chem.MolFromSmiles(smiles):
        mols.append(Chem.MolFromSmiles(smiles))
        names.append(name)
# print(len(mols))

calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])

test_desc = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in mols],
                           columns = list(calc.GetDescriptorNames()), index = names)
test_desc = test_desc.loc[:, ~test_desc.isnull().any()]
# print(test_desc.head())

test_desc_std = pd.DataFrame(scale(test_desc), index = test_desc.index, columns = test_desc.columns)
test_desc_std = test_desc_std.round(3)
# print(test_desc_std.shape)

''' Adding the principal components '''
pca = PCA(n_components = 3)

principalComponents = pca.fit_transform(descriptors_train_std.values)
# print(principalComponents)

pca_test = pca.fit_transform(test_desc_std.values) # remove this if pca is just for one set (training or test)
# print(pca_test.shape)

''' Plotting the activity wrt the principal components '''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for pts, act in zip(principalComponents, training["activity"].values):
    # if act == 1:
    #     color = 'red'
    # else:
    #     color = 'red'
    color = 'teal'

    ax.scatter(pts[0], pts[1], pts[2], color = color, alpha = 0.4)

for pts, cls in zip(pca_test, test["activity"].values):
    # if cls == 1:
    #     color = 'green'
    # else:
    #     color = 'green'
    color = 'mediumvioletred'
    ax.scatter(pts[0], pts[1], pts[2], color = color, alpha = 0.4)

ax.set(xlim=(-10,30), ylim=(-10, 30), zlim=(-15,8))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
ax.set_zlabel('PCA 3')


plt.show()

