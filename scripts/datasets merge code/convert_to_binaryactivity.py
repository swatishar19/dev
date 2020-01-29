import pandas as pd

file = pd.read_csv('toxref_all_DevTox.csv')

# print(list(file))
'''
bin_act = []
for i in range(len(file)):
    if file.iloc[i, -1] <=21:
        bin_act.append(1)
    elif file.iloc[i, -1] > 21:
        bin_act.append(0)

file['toxref activity'] = bin_act

file.to_csv('Toxref Developmental all species.csv')
'''
print(file.species.unique())