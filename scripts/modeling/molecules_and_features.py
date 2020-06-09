from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

from sklearn.metrics.pairwise import pairwise_distances

import pandas as pd
import os
import math

import matplotlib.pyplot as plt

def generate_molecules(dataset_name, data_dir=None, endpoint=None):
    """
    Takes in the path to a QSAR-ready .sdf file and generates a list of rdkit molecule objects
    Returns training and external validation sets without molecules that could not be created from SMILES strings or
    molecules with no information about the desired endpoint
    Returns prediction sets without molecules that could not be created from SMILES strings

    :param dataset_name: String representing dataset name in .sdf file
    :param data_dir: Environment variable pointing to the project directory containing the dataset
    :param endpoint: Desired binary or continuous endpoint with threshold to be modeled (for training sets and external
    validation sets, defaults to None); Enter None for prediction sets

    :return molecules: List of rdkit molecule objects
    """

    # Checks for appropriate input
    assert isinstance(dataset_name, str), 'The input parameter dataset_name (%s) must be a string.' % dataset_name
    assert isinstance(data_dir, str), 'The input parameter data_dir (%s) must be a string.'
    assert data_dir is not None, 'Please create an environment variable called (%s) pointing to the project directory ' \
                                'containing the dataset.' % data_dir

    # Instantiates data_dir and sdf_file variables
    sdf_file = os.path.join(data_dir, '{}.sdf'.format(dataset_name))

    # Checks for appropriate input
    assert os.path.exists(sdf_file), 'The dataset entered (%s) is not present in data_directory as a  .sdf file' \
                                     % dataset_name
    assert isinstance(endpoint, str) or endpoint is None, 'The input parameter endpoint (%s) must either be a string ' \
                                                          'or None.' % endpoint

    if endpoint is None:
        # Returns rdkit Mol objects for molecules in prediction set .sdf files that were able to be generated
        molecules = [mol for mol in Chem.SDMolSupplier(sdf_file) if mol is not None]

        # Checks for appropriate output
        assert molecules != [], 'No molecules could be generated from the dataset provided (%s).' % dataset_name

        return molecules

    else:
        # Returns rdkit Mol objects for molecules in training and evaluation set .sdf files that were able to be
        # generated and have information pertaining to the desired endpoint
        molecules = [mol for mol in Chem.SDMolSupplier(sdf_file) if mol is not None and mol.HasProp(endpoint)
                     and mol.GetProp(endpoint) not in ['NA', 'nan', '']]

        # Checks for appropriate output
        assert molecules != [], 'No molecules could be generated with the given endpoint (%s)' % endpoint

        return molecules


def calc_rdkit(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules, calculates molecular descriptors for each molecule, and returns a machine
    learning-ready pandas DataFrame.

    :param molecules: List of rdkit molecule objects with no None values
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    # Generates molecular descriptor calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])

    # Calculates descriptors and stores in pandas DataFrame
    X = pd.DataFrame([list(calculator.CalcDescriptors(mol)) for mol in molecules],
                     index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules],
                     columns=list(calculator.GetDescriptorNames()))

    # Imputes the data and replaces NaN values with mean from the column
    desc_matrix = X.fillna(X.mean())

    # Checks for appropriate output
    assert len(desc_matrix.columns) != 0, 'All features contained at least one null value. No descriptor matrix ' \
                                          'could be generated.'

    return desc_matrix


def calc_ecfp6(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns ECFP6 fingerprints for a list of rdkit molecules

    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecule objects with no None values

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)]
        data.append(ecfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_fcfp6(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns FCFP6 fingerprints for a list of rdkit molecules

    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecules with no None values

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        fcfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)]
        data.append(fcfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_maccs(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns MACCS fingerprints for a list of rdkit molecules

    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecules with no None values

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        maccs = [int(x) for x in MACCSkeys.GenMACCSKeys(mol)]
        data.append(maccs)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def get_activities(molecules, name_col='CASRN', endpoint=None, threshold=None, regress=False):
    """
    Takes in a list of rdkit molecules and returns a vector with a value of 1 if the indexed molecule fits the desired
    binary endpoint and 0 if it does not

    :param molecules: List of rdkit molecule objects with no None values
    :param name_col: Name of the field to index the resulting Series.  Needs to be a valid property of all molecules
    :param endpoint: Desired property to be modeled (defaults to None). Needs to be a valid property of all molecules
    :param threshold: Toxicity threshold value for binary endpoints based on continuous data where values falling
    below the threshold will constitute an active response and vice versa(i.e. LD50 in mg/kg, defaults to None)
    :param regress: True if continuous endpoint is to be modeled (e.g. LD50 in mg/kg), False otherwise (defaults to
    False)

    :return y: Activity vector as pandas Series
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The input entered is not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert all(mol.HasProp(endpoint) for mol in molecules), 'The desired endpoint is not valid for all molecules in ' \
                                                            'the input list.'
    assert all(mol.GetProp(endpoint) != '' and mol.GetProp(endpoint) != 'NA' for mol in molecules), 'The desired ' \
                                                                                                    'endpoint is not ' \
                                                                                                    'valid for all ' \
                                                                                                    'molecules in ' \
                                                                                                    'the input list.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col
    assert all(mol.HasProp(name_col) for mol in molecules), 'The input parameter name_col (%s) must be a valid ' \
                                                            'property of all molecules to be modeled.' % name_col
    assert endpoint is not None, 'Please enter a binary endpoint or continuous endpoint and threshold value.'
    assert isinstance(endpoint, str), 'The input parameter endpoint (%s) must be a string.' % endpoint

    # Instantiates activity vector
    y = []

    # Checks for binary endpoints
    if molecules[0].GetProp(endpoint).title() in ['True', 'False']:
        for mol in molecules:
            # Populates the activity vector with a 1 for molecules with a 'True' endpoint and vice versa
            y.append(int(eval(mol.GetProp(endpoint).title())))
        return pd.Series(y, index=[mol.GetProp(name_col) for mol in molecules])

    elif molecules[0].GetProp(endpoint) in ['0.0', '0', '1.0', '1']:

        for mol in molecules:
            y.append(int(float(mol.GetProp(endpoint))))

        return pd.Series(y, index=[mol.GetProp(name_col) for mol in molecules])

    # Populates the activity vector with a 1 for molecules with a continuous endpoint below the threshold and vice versa
    elif not regress:
        for mol in molecules:
            continuous_value = float(mol.GetProp(endpoint))
            if continuous_value < threshold:
                y.append(1)
            elif continuous_value >= threshold:
                y.append(0)
        return pd.Series(y, index=[mol.GetProp(name_col) for mol in molecules])

    elif regress:
        y_continuous = []
        for mol in molecules:
            continuous_value = float(mol.GetProp(endpoint))
            y_continuous.append(math.log10(continuous_value))
        return pd.Series(y_continuous, index=[mol.GetProp(name_col) for mol in molecules])

    else:
        raise Exception('Please enter a binary endpoint or continuous endpoint value (with or without a threshold '
                        'value.')


def load_external_desc(dataset_name, features, data_dir=None, pred_set=False, training_set=None, endpoint=None, threshold=None):
    """
    Loads externally generated descriptor values stored in .csv files saved to a sub-folder of data_dir titled
    'external_descriptors' with the format '(sdf_file)_descriptors.sdf'

    :param dataset_name: Name of the .csv file containing externally generated descriptor values
    :param descriptors: Molecular descriptor set to be used for modeling
    :param data_dir: Environmental variable pointing to the project directory

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    assert data_dir is not None, 'Please set an environmental variable pointing to your project directory to use this' \
                                 'function.'
    assert isinstance(data_dir, str), 'The input parameter data_dir (%s) must be a string.'
    assert os.path.exists(data_dir), 'Please create an environment variable called (%s) pointing to the ' \
                                                'project directory containing the dataset.' % data_dir
    file = os.path.join(data_dir, 'external_descriptors', '{}_{}.csv'.format(dataset_name, features))

    assert os.path.exists(file), "Externally generated descriptor values must be stored in .csv files saved to a " \
                                 "sub-folder of data_dir titled 'external_descriptors' with the format " \
                                 "'{}_{}.csv'.format(dataset_name, descriptors)"
    assert isinstance(features, str), 'The input parameter descriptors (%s) must be a string.' % features

    X = pd.read_csv(file, index_col=0)
    df = X.copy()
    df.loc[:, df.isnull().all()] = -999
    df.fillna(X.mean(), inplace=True)

    if pred_set:
        X_train, y_train = make_dataset('{}.sdf'.format(training_set), data_dir=os.getenv('NICEATM_ACUTE_ORAL_DATA'),
                                        pred_set=False, features=features, endpoint=endpoint, threshold=threshold)
        df = df[X_train.columns]

    return df


def get_classes(molecules, name_col='CASRN', class_col='Class'):
    return pd.Series([int(mol.GetProp(class_col)) for mol in molecules],
                     index=[mol.GetProp(name_col) if mol.HasProp(name_col) else ''
                            for mol in molecules])


def make_dataset(sdf_file, data_dir=None, pred_set=False, features='MACCS', name_col='CASRN', endpoint=None,
                 threshold=None, regress=False):
    """
    :param sdf_file: Name of the .sdf file from which to make a dataset
    :param data_dir: Environmental variable pointing to the project directory
    :param pred_set: True if the dataset is a prediction set, False otherwise (defaults to False)
    :param features: Molecular descriptor set to be used for modeling (defaults to MACCS keys); If externally
    generated descriptors are used, they must be saved to a sub-folder of data_dir titled 'external_descriptors' with
    the format 'datasetname_features.sdf'
    :param name_col: Name of the field to index the resulting Series.  Needs to be a valid property of all molecules
    :param endpoint: Desired property to be modeled (defaults to None). Needs to be a valid property of all
    molecules
    :param threshold: Toxicity threshold value for binary endpoints based on continuous data where values falling
    below the threshold will constitute an active response and vice versa(i.e. LD50 in mg/kg, defaults to None)
    :param regress: True if the model will be a regression model, False otherwise (defaults to False)

    If it is the first time the descriptors (and activities, when applicable) are being loaded, it will generate a
    descriptor matrix and activity vector for training and external validation sets and then cache the results to a
    .csv file for easier future loading. Descriptor matrices will be generated and cached for prediction sets.

    :return (X, y): (Feature matrix as a pandas DataFrame, Class labels as a pandas Series (Only returned for
    training and external validation sets))
    """

    # Checks for appropriate input
    assert sdf_file[-4:] == '.sdf', 'The input parameter sdf_file must be in .sdf file format.'
    assert data_dir is not None, 'Please set an environmental variable pointing to your project directory to use this' \
                                'function.'
    assert isinstance(pred_set, bool), 'The input parameter prediction_set (%s) must be a Boolean value ' \
                                             '(True/False)' % pred_set
    assert isinstance(features, str), 'The input parameter features (%s) must be a string.' % features
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col
    assert isinstance(endpoint, str) or endpoint is None, 'The input parameter endpoint (%s) must be a string or ' \
                                                          'None.' % endpoint
    assert isinstance(threshold, int or float) or threshold is None, 'The input parameter threshold (%s) must be an ' \
                                                                     'integer or floating point number for ' \
                                                                     'continuous data or None for binary data.' \
                                                                     % threshold

    if endpoint is None and not pred_set:
        raise Exception('Please enter a binary or continuous endpoint for modeling.')

    if regress and threshold is None:
        raise Exception('Please enter a threshold value to be used for data splitting during model cross-validation.')

    if endpoint == 'Class':
        threshold = 'X'

    descriptor_fxs = {
        'rdkit': lambda mols: calc_rdkit(mols, name_col=name_col),
        'ECFP6': lambda mols: calc_ecfp6(mols, name_col=name_col),
        'FCFP6': lambda mols: calc_fcfp6(mols, name_col=name_col),
        'MACCS': lambda mols: calc_maccs(mols, name_col=name_col)
    }

    dataset_name = sdf_file.split('.')[0]

    data_dir = os.getenv(data_dir)


    if not os.path.exists(os.path.join(data_dir, 'caches')):
        os.makedirs(os.path.join(data_dir, 'caches'))

    if features not in list(descriptor_fxs.keys()) + ['toxprint', 'dragon']:
        raise Exception('Sorry, that feature set is not yet available.\n'
                        'Available features are:\n{}'.format('\n'.join(descriptor_fxs.keys())))

    # return the previously cached file if it exists
    # if not use calc_descriptors and get_activities to make an X, y
    if threshold is not None:
        cache_file_dir = os.path.join(data_dir, 'caches', '{}_{}_{}_{}.csv'.format(dataset_name, features, endpoint,
                                                                                   threshold))
        cache_file_regress = os.path.join(data_dir, 'caches', '{}_{}_{}_{}_regression.csv'.format(dataset_name,
                                                                                                  features, endpoint,
                                                                                                  threshold))

    else:
        cache_file_dir = os.path.join(data_dir, 'caches', '{}_{}_{}.csv'.format(dataset_name, features, endpoint))
        cache_file_regress = os.path.join(data_dir, 'caches', '{}_{}_{}_regression.csv'.format(dataset_name,
                                                                                               features, endpoint))

    if not regress:
        if not pred_set and os.path.exists(cache_file_dir):
            df = pd.read_csv(cache_file_dir, index_col=0, encoding='ISO-8859-1', low_memory= False)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            return X, y

    else:
        print(cache_file_regress)
        if not pred_set and os.path.exists(cache_file_regress):
            df = pd.read_csv(cache_file_regress, index_col=0)
            X = df.iloc[:, :-2]
            y_continuous = df.iloc[:, -1]
            y_class = df.iloc[:, -2]

            return X, y_continuous, y_class

    # Creates X and y for external features
    if os.path.exists(os.path.join(data_dir, 'external_descriptors', '{}_{}.csv'.format(dataset_name, features))):
        molecules = generate_molecules(dataset_name, data_dir, endpoint)

        if regress:
            X, y_continuous, y_class = load_external_desc(dataset_name, features, data_dir), \
                                       get_activities(molecules, name_col, endpoint, threshold, regress), \
                                       get_activities(molecules, name_col, endpoint, threshold)

            assert y_class.index == y_continuous.index

            X = X[X.index.isin(y_class.index)]

            for first, second in zip(X.index, y_class.index):
                if first != second:
                    print(first, second)

            assert X.index == y_class.index

            df = X.copy()
            df['Class'] = y_class
            df['Continuous_Value'] = y_continuous
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}_{}_regression.csv'.format(dataset_name,
                                                                                           features, endpoint,
                                                                                           threshold)))

            return X, y_continuous, y_class


        elif threshold is not None:
            X, y = load_external_desc(dataset_name, features, data_dir), get_activities(molecules, name_col,
                                                                                        endpoint, threshold,
                                                                               regress)
            X = X.loc[y.index]
            df = X.copy()
            df['Class'] = y
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}_{}.csv'.format(dataset_name, features, endpoint,
                                                                                threshold)))

            return X, y

        if pred_set:
            if os.path.exists(os.path.join(data_dir, 'caches', '{}_{}_prediction_set.csv'.format(dataset_name,
                                                                                                 features))):

                return pd.read_csv(os.path.join(data_dir, 'caches', '{}_{}_prediction_set.csv'.format(dataset_name,
                                                                                                      features)),
                                   index_col=0)

            else:
                molecules = generate_molecules(dataset_name, data_dir, endpoint=None)
                X = load_external_desc(dataset_name, features, data_dir, pred_set=True,
                                       training_set='trainingset_171127', endpoint='LD50_mgkg', threshold=2000)
                df = X.copy()
                df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_prediction_set.csv'.format(dataset_name, features)))

                return X

        else:
            # changing the line of code here from get_act to get_classes 
            # because if im following the logic correctly anytime
            # threshold is none, it would be a categorical endpoint and should call get_classes
            X, y = load_external_desc(dataset_name, features, data_dir), get_classes(molecules, name_col, endpoint)
            X = X.loc[y.index]
            df = X.copy()
            df['Class'] = y
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}.csv'.format(dataset_name, features, endpoint)))

        return X, y

    if pred_set:
        if os.path.exists(os.path.join(data_dir, 'caches', '{}_{}_prediction_set.csv'.format(dataset_name,
                                                                                             features))):

            return pd.read_csv(os.path.join(data_dir, 'caches', '{}_{}_prediction_set.csv'.format(dataset_name,
                                                                                                  features)),
                               index_col=0)

        else:
            molecules = generate_molecules(dataset_name, data_dir, endpoint=None)
            X = descriptor_fxs[features](molecules)
            df = X.copy()
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_prediction_set.csv'.format(dataset_name, features)))

            return X

    elif regress:

        molecules = generate_molecules(dataset_name, data_dir, endpoint)
        X, y_continuous, y_class = descriptor_fxs[features](molecules), get_activities(molecules, name_col,
                                                                                       endpoint, threshold,
                                                                                       regress), get_activities(
            molecules, name_col, endpoint, threshold)

        df = X.copy()
        df['Class'] = y_class
        df['Continuous_Value'] = y_continuous
        df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}_{}_regression.csv'.format(dataset_name,
                                                                                       features, endpoint,
                                                                                       threshold)))

        return X, y_continuous, y_class

    elif endpoint not in ['Class', 'EPA_category', 'GHS_category']:
        molecules = generate_molecules(dataset_name, data_dir, endpoint)
        X, y = descriptor_fxs[features](molecules), get_activities(molecules, name_col, endpoint, threshold)

        df = X.copy()
        df['Class'] = y

        if threshold is not None:
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}_{}.csv'.format(dataset_name, features, endpoint,
                                                                                threshold)))

        else:
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}.csv'.format(dataset_name, features, endpoint)))

        return X, y

    else:
        molecules = generate_molecules(dataset_name, data_dir, endpoint=endpoint)
        X, y = descriptor_fxs[features](molecules), get_classes(molecules, name_col, endpoint)

        df = X.copy()
        df['Class'] = y

        if threshold is not None:
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}_{}.csv'.format(dataset_name, features, endpoint,
                                                                                threshold)))
        else:
            df.to_csv(os.path.join(data_dir, 'caches', '{}_{}_{}.csv'.format(dataset_name, features, endpoint)))

        return X, y

def calc_chem_similarity(dataset_name, data_dir=None, name_col='CASRN', endpoint=None, features='MACCS', set_2=None):
    X = make_dataset('{}.sdf'.format(dataset_name), data_dir, True, features, name_col, endpoint)
    X_2 = make_dataset('{}.sdf'.format(set_2), data_dir, True, features, name_col, endpoint)
    jaccard_similarities = pd.DataFrame(pairwise_distances(X, Y=X_2, metric='jaccard'), columns=X_2.index)

    if set_2 == None:
        jaccard_similarities.to_csv(os.path.join(os.getenv(data_dir), 'caches', 'chem_similarity', dataset_name + '_'
                                                 + features + '.csv'))

    else:
        jaccard_similarities.to_csv(
            os.path.join(data_dir, 'caches', 'chem_similarity', dataset_name + '_' + set_2 + '_' +
                         features + '.csv'))

    distance_to_nn = []
    nearest_neighbors = []

    for i, neighbors in jaccard_similarities.iterrows():
        smallest_distance = neighbors.sort_values(ascending=True).values[0]
        distance_to_nn.append(smallest_distance)
        nn = neighbors == smallest_distance
        nearest_neighbors.append(neighbors[nn].index.tolist())

    distances = pd.DataFrame(index=X.index)
    distances['Nearest Neighbor'] = nearest_neighbors
    distances['Distance'] = distance_to_nn
    distances.to_csv(os.path.join(data_dir, 'caches', 'chem_similarity', dataset_name + '_' + features +
                                  '_distances.csv'))
    distance_to_nn.sort()


    plt.figure(2)
    #plt.hist(distance_to_nn, 100, alpha=0.5)
    plt.plot([i for i in range(len(distance_to_nn))], distance_to_nn)
    plt.xlabel('Compounds')
    plt.ylabel('Jaccard Distance to Nearest Neighbor')
    plt.title('Jaccard Similarities')
    axes = plt.gca()
    axes.set_ylim([0, 1])

    if set_2 == None:
        plt.savefig(os.path.join(data_dir, 'caches', 'chem_similarity', dataset_name + '_'
                                                 + features + '.png'))

    else:
        plt.savefig(os.path.join(data_dir, 'caches', 'chem_similarity', dataset_name + '_'
                                 + set_2 + '_' + features + '.png'))


#make_dataset('trainingset_171127.sdf', env_var='NICEATM_ACUTE_ORAL_DATA', features='dragon', regress=True,
#             threshold=1000, endpoint='LD50_mgkg')


##if __name__ == '__main__':
##    dataset = 'trainingset'
##    model_endpoint = 'LD50_mgkg'
##    model_threshold = 1000
##    X_train, y_train = make_dataset('{}.sdf'.format(dataset), endpoint=model_endpoint, threshold=model_threshold)
