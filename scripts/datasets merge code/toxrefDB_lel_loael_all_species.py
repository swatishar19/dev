import pandas as pd

toxref = pd.read_csv("toxrefdb_nel_lel_noael_loael_summary_AUG2014_FOR_PUBLIC_RELEASE.csv", encoding='ISO-8859-1', low_memory = False)
toxref  = toxref.drop(toxref.columns[0], axis = 1)

toxref_data = toxref.loc[toxref['study_type'].isin(['DEV'])]
toxref_data = toxref_data[['chemical_casrn', 'chemical_name', 'species', 'effect_category', 'lel_dose', 'loael_dose', 'study_type' ]]
# print(toxref_data.head())
#print(set(toxref_data.species)) # {'hamster', 'mouse', 'mink', 'rat'} are the species

toxref_data = toxref_data.fillna('NULL')

''' Delete the rows that have both the LOAEL and LEL values as Null'''
to_drop_list = []
for item in range(len(toxref_data)):
    if toxref_data.iloc[item, 4] == 'NULL' and toxref_data.iloc[item, 5] == 'NULL':
        to_drop_list.append(item)

toxref_data = toxref_data.drop(toxref_data.index[to_drop_list])

# print(set(toxref_data.species)) # {'hamster', 'mouse', 'mink', 'rat'} are the species

''' Only keep the minimum LEL/LOAEL value for each compound for MGR all species.'''
i = 0
j = 1

min_loael_lel_list = []
while j < len(toxref_data):

    if toxref_data.iloc[i, 0] == toxref_data.iloc[j, 0]: # 0 is chemical_casrn
        lowest_val_lst = []
        if toxref_data.iloc[i, 4] != 'NULL':
            lowest_val_lst.append(toxref_data.iloc[i, 4])
        if toxref_data.iloc[i, 5] != 'NULL':
            lowest_val_lst.append(toxref_data.iloc[i, 5])
        if toxref_data.iloc[j, 4] != 'NULL':
            lowest_val_lst.append(toxref_data.iloc[j, 4])
        if toxref_data.iloc[j, 5] != 'NULL':
            lowest_val_lst.append(toxref_data.iloc[j, 5])

        lowest_lel_loael = min(lowest_val_lst)

        if lowest_lel_loael == toxref_data.iloc[i, 4] or lowest_lel_loael == toxref_data.iloc[i, 5]: # 4 is lel_dose, 5 is loael_dose
            j+=1

        elif lowest_lel_loael == toxref_data.iloc[j, 4] or lowest_lel_loael == toxref_data.iloc[j, 5]:
            i = j
            j+=1

        else:
            print(min(toxref_data.iloc[i, 4], toxref_data.iloc[i, 5], toxref_data.iloc[j, 4], toxref_data.iloc[j, 5]) == toxref_data.iloc[i, 5], "SOMETHING'S WRONG", )

    elif toxref_data.iloc[i, 0] != toxref_data.iloc[j, 0]:
        min_loael_lel_list.append(toxref_data.iloc[i, :])
        i = j
        j+=1

    else:
        print('not working')

# print(min_loael_lel_list[0:30])


toxref_min_loael_lelDF = pd.DataFrame.from_records(min_loael_lel_list, columns=toxref_data.columns)

toxref_min_loael_lelDF.to_csv('toxref_allsp_DEV_minLelOrLoael.csv')

