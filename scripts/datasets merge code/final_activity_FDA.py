import pandas as pd

toxref_all = pd.read_csv('dev_fda_test.csv', encoding='ISO-8859-1', low_memory = False)

toxref_all = toxref_all[[ 'CAS_RN', 'NAME', 'rdkit SMILES', 'ACTIVITY_nbrat', 'ACTIVITY_fdrabbit', 'ACTIVITY_fdmouse', 'ACTIVITY_fdrat', 'ACTIVITY_nbmouse']]

toxref_all = toxref_all.fillna('NULL')

final_activity_list = []
for i in range(len(toxref_all)):
    activity_list = []
    if toxref_all.iloc[i, 3] != 'NULL':    #4 is 'lel_dose' from DEV study from toxref_nel* file
        activity_list.append(toxref_all.iloc[i, 3])
    if toxref_all.iloc[i, 4] != 'NULL':    #4 is 'lel_dose' from DEV study from toxref_nel* file
        activity_list.append(toxref_all.iloc[i, 4])
    if toxref_all.iloc[i, 5] != 'NULL':
        activity_list.append(toxref_all.iloc[i, 5]) #5 is 'loael_dose' from DEV study from toxref_nel* file
    if toxref_all.iloc[i, 6] != 'NULL':
        activity_list.append(toxref_all.iloc[i, 6])
    if toxref_all.iloc[i, 7] != 'NULL':
        activity_list.append(toxref_all.iloc[i, 7])



    final_activity_list.append(max(activity_list))

# print(len(final_activity_list), len(toxref_all))

toxref_all['FinalFDAactivity'] = final_activity_list

toxref_all.to_csv('fda_dev_testset.csv')