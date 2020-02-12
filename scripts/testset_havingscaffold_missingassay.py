import pandas as pd
from rdkit import Chem

training = pd.read_csv('curated dev_allspecies shengde,fda,caeser,toxref.csv',  encoding = 'ISO-8859-1', low_memory= False)
training_profile = pd.read_csv('filtered_trainingset_DEV_shengdeFDAtoxCAESER_bioprofile_woRemovingCorrelatedAssays.csv', encoding = 'ISO-8859-1', low_memory= False)

testset = pd.read_csv('curated_fdaTestset_devtox_curatedwithoutOverlaps4.csv', encoding = 'ISO-8859-1', low_memory= False)
testprofile = pd.read_csv('filtered_testset_profile_removed_all_overrlaps.csv', encoding = 'ISO-8859-1', low_memory= False)

scaffold = Chem.MolFromSmarts('[*]-[#6](=[*])-[#7](-[*])-[#6](@-[*])-@[*]')
def scaffold_search(training, testprofile,  scaffold, AID):
    chem = []
    for z in range(len(training)):
        m = Chem.MolFromSmiles(training.loc[z, 'SMILES'])
        if m.HasSubstructMatch(scaffold):
            chem.append(training.loc[z, ['Index', 'SMILES', 'activity']])
    chem_withscaffold = pd.DataFrame.from_records(chem, columns=['Index', 'SMILES', 'activity'])
    training_scaff_assay = pd.merge(chem_withscaffold, training_profile, on=['Index'], how='left')
    training_scaff_assay = training_scaff_assay[['Index', 'SMILES', AID, 'activity']]
    training_scaff_assay = training_scaff_assay[training_scaff_assay[AID] == 0] # just retrieving inconclusive assay

    test = testprofile[['CAS_RN', 'SMILES', AID, 'activity']]
    test = test[test[AID] == 0]

    chem2 = []
    for i in range(len(test)):
        m = Chem.MolFromSmiles(test.iloc[i, 1])
        if m.HasSubstructMatch(scaffold):
            chem2.append(test.iloc[i, :])
    test_withscaffold = pd.DataFrame.from_records(chem2, columns=['CAS_RN', 'SMILES', AID, 'activity'])
    test_withscaffold = test_withscaffold.rename(columns={'CAS_RN': 'Index'})

    TestSetForFillingMissingData = pd.concat([training_scaff_assay, test_withscaffold])
    TestSetForFillingMissingData = TestSetForFillingMissingData.rename(columns={'Index':'Name', 'activity':'DevtoxActivity', AID: 'Activity'})
    print(len(TestSetForFillingMissingData))
    TestSetForFillingMissingData.to_csv('testset_pubchem_missingdata_model_1347031_Nunknown.csv')

scaffold_search(training, testprofile, scaffold, AID = '1347031')


