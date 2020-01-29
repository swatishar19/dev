from urllib.request import urlopen
import pandas as pd


def CAS_to_SMILES(filename, cas_column):
    df = pd.read_csv(filename + '.csv', encoding='ISO-8859-1', low_memory = False)

    smiles = []
    cas_list = []
    for i in range(len(df)):

        try:
            url = 'http://cactus.nci.nih.gov/chemical/structure/' + df.loc[i, cas_column] + '/smiles'
            ans = urlopen(url).read().decode('utf8')
            smiles.append(ans)
        except:
            smiles.append('##### """SMILES not found""" #####')

    df['SMILES'] = smiles

    df.to_csv(filename + '_SMILES_added.csv')


CAS_to_SMILES('ToxRefDB_endpointMatrix_MGRratDEVREP', 'chemical_casrn')
