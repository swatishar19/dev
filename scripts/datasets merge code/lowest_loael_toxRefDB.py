import pandas as pd
nel_file = pd.read_csv("toxrefdb_nel_lel_noael_loael_summary_AUG2014_FOR_PUBLIC_RELEASE.csv", encoding='ISO-8859-1', low_memory = False)
nel_file = nel_file.drop(nel_file.columns[0], axis = 1)

nellel_data = nel_file.loc[nel_file['study_type'].isin(['MGR', 'REP'])]
nellel_data = nellel_data[['chemical_casrn', 'chemical_name', 'species', 'effect_category', 'lel_dose', 'loael_dose', 'study_type' ]]
#print(list(nellel_data))
nellel_data = nellel_data.dropna(subset = ['loael_dose'])
#print(nellel_data.head())

#nellel_data.to_csv('loael.csv')
#print(nellel_data.columns.get_loc('lel_dose'))


i = 0
j = 1
loael_list = []
while ( j < len(nellel_data)):
    if nellel_data.iloc[i, 0] == nellel_data.iloc[j,0 ] and nellel_data.iloc[i, 6] == nellel_data.iloc[j, 6]: # 0 is casrn and 6 is study type

        if nellel_data.iloc[i,5] < nellel_data.iloc[j,5]: # 5 is loael_dose
            j +=1
        elif nellel_data.iloc[i, 5] > nellel_data.iloc[j, 5]:
            i = j
            j += 1
        elif nellel_data.iloc[i, 5] == nellel_data.iloc[j, 5]:
            j+=1

    elif nellel_data.iloc[i, 0] == nellel_data.iloc[j, 0] and nellel_data.iloc[i, 6] != nellel_data.iloc[j, 6]:
        loael_list.append(nellel_data.iloc[i,:])
        i=j
        j+=1
    elif nellel_data.iloc[i, 0] != nellel_data.iloc[j, 0]:
        loael_list.append(nellel_data.iloc[i,:])
        i=j
        j+=1
loael_list.append(nellel_data.iloc[i,:])

toxref_min_loaelDF = pd.DataFrame.from_records(loael_list, columns=nellel_data.columns)
toxref_min_loaelDF.to_csv('toxref_rep_minloael_rat_mouse.csv')
#print(len(cas_study_min_loaelDF))
#print(cas_study_min_loaelDF.head())
#print(toxref_min_loaelDF['species'])
'''
activ_list = []
for i in range(0, len(cas_study_min_loaelDF)):
    if cas_study_min_loaelDF.loc[i, 'loael_dose'] > 30:
        activ_list.append(0)
    else:
        activ_list.append(1)
#print(len(nellel_data))

cas_study_min_loaelDF['Activity'] = activ_list
#print(cas_study_min_loaelDF.loc[:, 'Activity'])
#print(cas_study_min_loaelDF.head())

cas_study_min_loaelDF = cas_study_min_loaelDF[['chemical_id', 'chemical_casrn', 'chemical_name', 'ldt', 'hdt', \
                                               'dose_unit', 'study_type', 'species', 'effect_category', 'lel_dose_level', \
                                               'lel_dose', 'nel_dose', 'loael_dose', 'noael_dose', 'Activity']]
#print(cas_study_min_loaelDF.head())
#cas_study_min_loaelDF.to_csv('activity_added_loael.csv')

#print(cas_study_min_loaelDF.columns.get_loc('study_type'))
#print(cas_study_min_loaelDF.columns.get_loc('species'))

DEV_rat_list = []
for i in range(len(cas_study_min_loaelDF)):
    if cas_study_min_loaelDF.iloc[i, 6] == 'DEV' and cas_study_min_loaelDF.iloc[i, 7] == 'rat':
        DEV_rat_list.append(cas_study_min_loaelDF.iloc[i, :])

DEV_rat_DF = pd.DataFrame.from_records(DEV_rat_list, columns = cas_study_min_loaelDF.columns)
print(DEV_rat_DF.head())


DEV_rat_DF.to_excel('DEVrat.xlsx')

#species_count = cas_study_min_loaelDF['study_type'].value_counts()
#print(species_count)


compSmiles = pd.read_csv('DEVrabbitCompounds_CIIPro.txt', sep="\t", header=None)
compSmiles = compSmiles.drop(compSmiles.columns[2], axis = 1)
compSmiles = compSmiles.drop(compSmiles.columns[0], axis = 0)
#compSmiles = compSmiles.drop(compSmiles.columns[1], axis = 0)
#print(compSmiles.head())


smiles = []
for i in range(1, len(compSmiles)):
    smiles.append(compSmiles.iloc[i, 2])
#print(smiles)

#print(compSmiles.iloc[:, 2])

DEV_rabbit_DF['SMILES'] = smiles
print(DEV_rabbit_DF.head())
DEV_rabbit_DF.to_excel('RabbitDevTox.xlsx')
#print(DEV_rabbit_DF.iloc[165, :])


'''
