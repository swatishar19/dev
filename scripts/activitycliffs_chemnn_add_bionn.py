import pandas as pd

chemnn = pd.read_csv('chem nn with activity_allCompd.csv')

chem_actcliffs = chemnn[chemnn['compound C activity'] != chemnn['compound D activity']]
# print(chem_actcliffs.head())

df = chem_actcliffs.groupby('compound C').apply(lambda x: x.sort_values('chemical_similarity', ascending = False))

df_max_similarity = df.drop_duplicates(subset = 'compound C', keep = 'first')
# print(df_max_similarity.head(20))
# print(df.head(20))
# print(len(df_max_similarity))
# df_max_similarity.to_csv('chemical nn activity cliffs with most similarity.csv')


######## Adding biosimilar nearest neighbors information to chemical nearest neighbors ########

bionn = pd.read_csv('bio nn with activity_allCompd.csv')

df_biosim = bionn.groupby('compound A').apply(lambda x: x.sort_values('Confidence', ascending = False))

# print(df_biosim.head(20))
df_maxBiosim = df_biosim.drop_duplicates(subset = 'compound A', keep = 'first')
# df_maxBiosim.to_csv('biosimilar nn with max confidence.csv')

# print(list(df_max_similarity), "\n")
# print(list(df_maxBiosim))

bionn = ['null'] * len(df_max_similarity)
bionn_act = ['null'] * len(df_max_similarity)
conf_list = ['null'] * len(df_max_similarity)
biosim_list = ['null'] * len(df_max_similarity)

for i in range(len(df_max_similarity)):
    for j in range(len(df_maxBiosim)):
        if df_max_similarity.iloc[i, 2] == df_maxBiosim.iloc[j, 4]:
            bionn[i] = df_maxBiosim.iloc[j, 5]
            bionn_act[i] = df_maxBiosim.iloc[j, 7]
            conf_list[i] = df_maxBiosim.iloc[j, 2]
            biosim_list[i] = df_maxBiosim.iloc[j, 3]

# print(len(bionn), len(bionn_act), len(df_max_similarity))
df_chemnn_bionn = df_max_similarity.copy()
df_chemnn_bionn['bionearest neighbor'] = bionn
df_chemnn_bionn['bio nn activity'] = bionn_act
df_chemnn_bionn['Confidence'] = conf_list
df_chemnn_bionn['biosimilarity'] = biosim_list

print(df_chemnn_bionn.head(20))

df_chemnn_bionn.to_csv('chem nn and bio nn with activities.csv')