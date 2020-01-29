import pandas as pd

caeser = pd.read_csv('fda NB classifier data.csv', encoding='ISO-8859-1', low_memory = False)
# caeser = caeser.drop(caeser.columns[0], axis =1)
# print(list(caeser))


bin_act_list = []
for i in range(len(caeser)):
    if caeser.iloc[i, 2] == 'Non-toxicant':
        bin_act_list.append(0)
    elif caeser.iloc[i, 2] == 'Developmental toxicant':
        bin_act_list.append(1)

print(len(bin_act_list))
caeser['FDA_activity'] = bin_act_list

caeser.to_csv('FDA_NBclassifier_withActivity.csv')