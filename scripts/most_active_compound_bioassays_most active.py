import pandas as pd
from collections import OrderedDict

def compounds_with_most_activities(file, compd = 10): #10 is for top 10 compounds with most activities
    dict_index_compounds = {}
    list_columns = list(file)
    for i in range(len(file)):
        count_active = 0
        for j in range(len(list_columns)):
            if file.iloc[i,j] == 1 or file.iloc[i,j] == -1:
                count_active +=1
        dict_index_compounds[file.iloc[i,0]] = count_active
        ordered_dict = OrderedDict(sorted(dict_index_compounds.items(), key = lambda t:t[1], reverse = True))
        list_ordered_dict = list(ordered_dict.items())
    return list_ordered_dict[:compd]  


df = pd.read_csv('filtered_training_devtox_shengdeFDAtoxCAESER_bioprofile_compdNames.csv', sep = ',', encoding='ISO-8859-1', low_memory= False)

# df = df.set_index(df.columns[0])
# print(df.iloc[0,0])
top_10 = compounds_with_most_activities(df)
print(top_10)

def assay_with_most_compounds(file, assay=10): # top 10 assays with most compounds information
    dict_bioassay_index = {}
    list_columns_bioassays = list(file)
    for j in range(len(list_columns_bioassays)):
        count_cmpd_act = 0
        for i in range(len(file)):
            if file.iloc[i, j] == 1 or file.iloc[i, j] == -1:
                count_cmpd_act +=1
        dict_bioassay_index[list_columns_bioassays[j]] = count_cmpd_act
        ordered_dict = OrderedDict(sorted(dict_bioassay_index.items(), key = lambda t:t[1], reverse = True))
        list_ordered_bioassay = list(ordered_dict.items())
    return list_ordered_bioassay[:assay]

# df = pd.read_csv('filtered_trainingset_devtox_shengdeFDAtoxCAESER_for_bioprofile.csv', sep = ',', encoding='ISO-8859-1', low_memory= False)
# df = df.set_index(df.columns[0])
# top_10_assays = assay_with_most_compounds(df)
# print(top_10_assays)

### confirming the above functions ###
# df = pd.read_csv('filtered_trainingset_devtox_shengdeFDAtoxCAESER_for_bioprofile.csv', sep = ',', encoding='ISO-8859-1', low_memory= False)
# print(df['1224869'].value_counts(), df['1259383'].value_counts(), df['1346978'].value_counts(), df['1346980'].value_counts())
#
#
# df = df.set_index(df.columns[0])
# print(df.iloc[517,].value_counts())#, df.iloc[35,].value_counts(), df.iloc[901,].value_counts(), df.iloc[517,].value_counts())


