import pandas as pd

toxref_matrix = pd.read_csv('toxrefdb_endpoint_matrix_Dev_all.csv', encoding='ISO-8859-1', low_memory = False)
toxref_matrix = toxref_matrix.drop(toxref_matrix.columns[0], axis =1)

toxref_matrix["X"] = toxref_matrix["X"].str.split('|')

casrn = []
name = []
for i in range(0, len(toxref_matrix)):
    min_num_lel = []
    casrn.append(toxref_matrix.loc[i, 'X'][1])
    name.append(toxref_matrix.loc[i, 'X'][2])

toxref_matrix['chemical_casrn'] = casrn
toxref_matrix['chemical_name'] = name

# print(len(toxref_matrix), 'before dropping na')
# toxref_matrix = toxref_matrix.dropna()

lel_value = []
for i in range(0, len(toxref_matrix)):
    temp = toxref_matrix.iloc[i, 1]
    for j in range(1, len(list(toxref_matrix))-2):
        if toxref_matrix.iloc[i, j] < temp:
            temp = toxref_matrix.iloc[i, j]
    lel_value.append(temp)

# print(list(toxref_matrix)[-3])


toxref_matrix['LEL DEV value from matrix'] = lel_value

rows_to_delete = []   # to delete the 1000000 values as they are
for item in range(len(toxref_matrix)):
    if toxref_matrix.iloc[item, -1 ] == 1000000:
        rows_to_delete.append(item)

toxref_matrix = toxref_matrix.drop(toxref_matrix.index[rows_to_delete])
# print(toxref_matrix.iloc[1, -1])


toxref_matrix['LEL DEV value from matrix'] = toxref_matrix['LEL DEV value from matrix'].fillna('NULL')

drop_null = []
for cmpd in range(len(toxref_matrix)):
    if toxref_matrix.iloc[cmpd, -1] == 'NULL':
        drop_null.append(cmpd)
toxref_matrix = toxref_matrix.drop(toxref_matrix.index[drop_null])
# print(len(toxref_matrix), 'after deleting 1 million and Null lel')
toxref_matrix.to_csv('ToxRefDB_endpointMatrix_DEV_all_species.csv')