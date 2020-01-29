import pandas as pd

devtox_all = pd.read_csv('dev_toxrefDB_caeser_fda_shengde.csv', encoding='ISO-8859-1', low_memory = False)

print(list(devtox_all))
devtox_all = devtox_all[[ 'chemical_casrn', 'chemical_name', 'Developmental Toxicity Effect', 'CAESAR v2.1.4\r\r\r\r\n(VEGA NIC v1.06)', 'rdkit SMILES', \
                          'shengde binary activity', 'FinalToxrefEndpoint', 'toxref activity', 'CAESAR class', 'caeser_activity', \
                          'Binary classification of teratogenicity', 'FDA classification', 'FDA_activity']]

devtox_all = devtox_all.fillna('NULL')

final_activity_list = []
for i in range(len(devtox_all)):
    activity_list = []
    if devtox_all.iloc[i, 5] != 'NULL':    #5 is 'shengde binary activity' from DEV study from shengde paper
        activity_list.append(devtox_all.iloc[i, 5])
    if devtox_all.iloc[i, 7] != 'NULL':    #7 is 'toxref activity' from DEV study from toxref
        activity_list.append(devtox_all.iloc[i, 7])
    if devtox_all.iloc[i, 9] != 'NULL':
        activity_list.append(devtox_all.iloc[i, 9]) #9 is 'caeser_activity'
    if devtox_all.iloc[i, 12] != 'NULL':
        activity_list.append(devtox_all.iloc[i, 12]) #12 is 'FDA_activity'

    final_activity_list.append(max(activity_list))

# print(len(final_activity_list), len(toxref_all))

devtox_all['final toxref, fda, caeser, shangde activity'] = final_activity_list

devtox_all.to_csv('DEVTOX combined toxref caeser fda shengde activity.csv')