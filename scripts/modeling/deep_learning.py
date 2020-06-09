import os
from sklearn import model_selection, utils

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, Nadam
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras.utils import np_utils

from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.externals import joblib
from sklearn.metrics import make_scorer

from molecules_and_features import make_dataset


def model_DNN_classifier(nh_layers=3, input_dim=196, num_hidden=[196, 196, 196],
                         num_labels=1, drop_out=0.1, beta=0.001, l_rate=0.01, momentum=0.9,
                         init_mode='he_normal', optimizer='SGD', activation='relu', activation_out='sigmoid',
                         model_summary=True):
    """
    N layers DNN model with drop out layers for each hidden layer. All the default parameters are for
    a 3 layer DNN model
    input:
    nh_layers: number of hidden layers
    input_dim: number of features in the dataset
    num_hidden: hidden layers size as a list
    num_labels: number of labels
    drop_out: the same drop out applied for all hidden layers
    beta: l2 regularization
    l_rate: initial learning rate
    momentum: used only for SGD optimizer
    init_mode: weight initiation method
    optimizer: optimizer name
    activation: hidden layers activation function
    activation_out: out layer activation function
    device: gpu:0 (can accept cpu:0 or gpu:1)
    odel_summary: show the model suumary, default is True)
    output:
    model: keras DNN model

    """
    if len(num_hidden) != nh_layers:
        return print("The number of layers nh_layers are not matched to the length of num_hidden")
    if activation == 'relu':
        act = 'relu'
    elif activation == 'tanh':
        act = 'tanh'
    elif activation == 'LeakyReLU':
        act = LeakyReLU()
    else:
        return print("I can't use this activation function to compile a DNN")

    model = Sequential(name='model_' + str(nh_layers) + '_layers')
    model.add(Dense(output_dim=num_hidden[0], kernel_initializer=init_mode, input_dim=input_dim,
                    W_regularizer=l2(l=beta), name='Dense_1'))
    model.add(Activation(act))
    model.add(Dropout(drop_out, name='DropOut_1'))
    for idx in range(nh_layers - 1):
        model.add(Dense(output_dim=num_hidden[idx + 1], kernel_initializer=init_mode,
                        W_regularizer=l2(l=beta), name='Dense_' + str(idx + 2)))
        if activation == 'LeakyReLU':
            act = LeakyReLU(input_shape=(num_hidden[idx + 1],))

        model.add(Activation(act))
        model.add(Dropout(drop_out, name='DropOut_' + str(idx + 2)))
    model.add(Dense(output_dim=num_labels, kernel_initializer=init_mode,
                    activation=activation_out, W_regularizer=l2(l=beta), name='Output'))
    # compile model
    if optimizer == 'SGD':
        opt = SGD(lr=l_rate, momentum=momentum, nesterov=True)
    elif optimizer == 'Adam':
        opt = Adam(lr=l_rate)
    elif optimizer == 'Nadam':
        opt = Nadam(lr=l_rate)
    else:
        return print("I don't know the %s optimizer" % optimizer)
    if model_summary:
        model.summary()
    model.compile(loss='binary_crossentropy', metrics=['binary_crossentropy', f1_score_k, 'accuracy'], #removed fmeasure
                  optimizer=opt)
    return model


class BatchLogger(Callback):
    def __init__(self, display):
        '''
        display: display progress every 5%
        '''
        #Callback.__init__(self)
        self.seen = 0
        self.display = display


def f1_score_k(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def cw_to_dict(y_class):
    """
    input: 1D array, labels
    output: balanced class weight dictionary
    """
    cw = utils.compute_class_weight('balanced', [0, 1], y_class) #compute class weight
    cw_dict = {}
    for idx in range(len(cw)):
        cw_dict[idx] = cw[idx]
    return cw_dict


def get_class_stats(model, X, y):
    """
    
    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return: 
    """
    if not model:
        predicted_classes = y
        predicted_probas = y
        y = X
    else:
        if 'predict_classes' in dir(model):
            predicted_classes = model.predict_classes(X, verbose=0)[:, 0]
            predicted_probas = model.predict_proba(X, verbose=0)[:, 0]
        else:
            predicted_classes = model.predict(X)
            predicted_probas = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, predicted_classes)
    f1_sc = f1_score(y, predicted_classes)

    # Sometimes SVM spits out probabilties with of inf
    # so set them as 1
    from numpy import inf
    predicted_probas[predicted_probas == inf] = 1

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y, predicted_probas)
    roc_auc = auc(fpr_tr, tpr_tr)
    # test classification results

    cohen_kappa = cohen_kappa_score(y, predicted_classes)
    matthews_corr = matthews_corrcoef(y, predicted_classes)
    precision = precision_score(y, predicted_classes)
    recall = recall_score(y, predicted_classes)

    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y, predicted_classes).ravel()
    specificity = tn / (tn + fp)
    ccr = (recall + specificity)/2
    
    return {'ACC': acc, 'F1-Score': f1_sc, 'AUC': roc_auc, 'Cohen\'s Kappa': cohen_kappa,
            'MCC': matthews_corr, 'Precision/PPV': precision, 'Recall': recall, 'Specificity': specificity, 'CCR': ccr}


def multiclass_model(nodes=1024, classes=5):
    beta = 0.001
    input_layer = Input(shape=(nodes,), name='input')
    dropout_1 = Dropout(0.1)(input_layer)
    first_layer = Dense(nodes, activation='relu', kernel_regularizer=l2(l=beta), kernel_initializer='he_normal')(
        dropout_1)
    dropout_2 = Dropout(0.1)(first_layer)
    second_layer = Dense(nodes, activation='relu', kernel_regularizer=l2(l=beta), kernel_initializer='he_normal')(
        dropout_2)
    dropout_3 = Dropout(0.1)(second_layer)
    third_layer = Dense(nodes, activation='relu', kernel_regularizer=l2(l=beta), kernel_initializer='he_normal')(
        dropout_3)
    dropout_4 = Dropout(0.1)(third_layer)

    output = Dense(classes, activation='softmax', name='output')(dropout_4)

    model = Model(inputs=[input_layer], outputs=output)
    model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'], optimizer='SGD')
    model.summary()
    return model


def regress_model(nodes=1024):
    beta = 0.001
    input_layer = Input(shape=(nodes,), name='input')
    dropout_1 = Dropout(0.1)(input_layer)
    first_layer = Dense(nodes, activation='relu', kernel_regularizer=l2(l=beta), kernel_initializer='he_normal')(
        dropout_1)
    dropout_2 = Dropout(0.1)(first_layer)
    second_layer = Dense(nodes, activation='relu', kernel_regularizer=l2(l=beta), kernel_initializer='he_normal')(
        dropout_2)
    dropout_3 = Dropout(0.1)(second_layer)
    third_layer = Dense(nodes, activation='relu', kernel_regularizer=l2(l=beta), kernel_initializer='he_normal')(
        dropout_3)
    dropout_4 = Dropout(0.1)(third_layer)

    output = Dense(1, name='output')(dropout_4)

    model = Model(inputs=[input_layer], outputs=output)
    model.compile(loss='mean_squared_error', metrics=['mean_squared_error', 'accuracy'], optimizer='SGD')
    model.summary()
    return model


def get_regress_stats(model, X, y):
    """

    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return:
    """
    if not model:
        predictions = y
        y = X
    else:
        if 'predict_classes' in dir(model):
            predictions = model.predict(X, verbose=0)[:, 0]
        else:
            predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {'MAE': mae, 'r2': r2}


def make_keras_pipe(num_steps, nh_layers, num_hidden, nodes, batch_size):
    """ Returns a ready-to-fit Keras Classifier pipeline
    
    :param num_steps: number of epochs to train data on 
    :param batch_size: batch size to put training through
    :return: 
    """
    out_batch = BatchLogger(display=5)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=50, min_lr=0.00001, verbose=1)
    stopping = EarlyStopping(monitor='loss', min_delta=0.0, patience=200, verbose=1, mode='auto')

    def new_fx():
        return model_DNN_classifier(nh_layers=nh_layers, num_hidden=num_hidden, input_dim=nodes)
    mdl_fx = new_fx

    model = KerasClassifier(mdl_fx, epochs=num_steps,
                            callbacks=[reduce_lr, stopping, out_batch],
                            batch_size=batch_size,
                            #class_weight=cw_tr_dict,
                            verbose=0)

    pipe = pipeline.Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)])
    return pipe


def save_model_pipe(pipe, n_layer, dataset, features, endpoint, threshold, save_dir):
    model_step = pipe.steps.pop(-1)[1]
    joblib.dump(pipe, os.path.join(save_dir,
                                   'DNN_{}_{}_{}_{}_{}_pipeline.pkl'.format(n_layer, dataset, features, endpoint,
                                                                            threshold)))
    models.save_model(model_step.model, os.path.join(save_dir,
                                                     'DNN_{}_{}_{}_{}_{}_model.h5'.format(n_layer, dataset, features,
                                                                                          endpoint, threshold)))


def load_keras_model(directory, name):
    pipe = joblib.load(os.path.join(directory, name + '_pipeline.pkl'))
    model = models.load_model(os.path.join(directory, name + '_model.h5'), custom_objects={'f1_score_k':f1_score_k})
    pipe.steps.append(('clf', model))
    return pipe


def save_multiclass_model_pipe(pipe, model, dataset, features, endpoint, threshold, save_dir):
    joblib.dump(pipe, os.path.join(save_dir,
                                   'DNN_{}_{}_{}_{}_{}_pipeline.pkl'.format('multiclass', dataset, features, endpoint,
                                                                            threshold)))
    models.save_model(model, os.path.join(save_dir,
                                          'DNN_{}_{}_{}_{}_{}_model.h5'.format('multiclass', dataset, features,
                                                                               endpoint, threshold)))


def load_multiclass_keras_model(directory, name):
    pipe = joblib.load(os.path.join(directory, name + '_pipeline.pkl'))
    model = models.load_model(os.path.join(directory, name + '_model.h5'), custom_objects={'f1_score_k':f1_score_k})

    return pipe, model


# scoring dictionary, just a dictionary containing the evaluation metrics passed through a make_scorer()
# fx, necessary for use in GridSearchCV

class_scoring = {'ACC': make_scorer(accuracy_score), 'F1-Score': make_scorer(f1_score), 'AUC': make_scorer(roc_auc_score),
           'Cohen\'s Kappa': make_scorer(cohen_kappa_score), 'MCC': make_scorer(matthews_corrcoef),
           'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score)}

regress_scoring = {'MAE': make_scorer(mean_absolute_error), 'r2_score': make_scorer(r2_score)}

if __name__ == '__main__':

    # import argparse



    # setting up GPU environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set GPU 0


    # parser = argparse.ArgumentParser(description='Utility to build Classic ML models')
    # parser.add_argument('-i', '--infile', required=True)
    # parser.add_argument('-nl', '--n_layer', required=True)
    # args = parser.parse_args()


    dataset = 'trainingset'
    features = 'ECFP6'
    n_layer = 3
    n_split = 5

    X_train, y_train = make_dataset('{}.sdf'.format(dataset), features)


    print("Training set includes {} descriptors".format(X_train.shape[1]))
    print("Training set includes {}, {} molecules".format(X_train.shape[0], y_train.shape[0]))


    nodes = X_train.shape[1]
    num_hidden = [nodes for _ in range(n_layer)]
    num_steps = 10001

    out_batch = BatchLogger(display=5)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=50, min_lr=0.00001, verbose=1)
    stopping = EarlyStopping(monitor='loss', min_delta=0.0, patience=200, verbose=1, mode='auto')
    batch_size = int(y_train.shape[0] // n_split)
    cw_tr_dict = cw_to_dict(y_train)

    def new_fx():
        return model_DNN_classifier(num_hidden=num_hidden, input_dim=nodes)
    mdl_fx = new_fx

    model = KerasClassifier(mdl_fx, epochs=num_steps,
                            callbacks=[reduce_lr, stopping, out_batch],
                            batch_size=batch_size,
                            #class_weight=cw_tr_dict,
                            verbose=0)

    pipe = pipeline.Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)])

    pipe.fit(X_train.values, y_train.values)

    stats = get_class_stats(pipe, X_train.values, y_train.values)

    print("Score for 5-fold CV:")
    for stat, val in stats.items():
        print("{}: {}".format(stat, val))

    save_dir = os.path.join(os.getenv('NICEATM_ACUTE_ORAL_DATA'), 'DL_models')

    model_step = pipe.steps.pop(-1)[1]
    joblib.dump(pipe, os.path.join(save_dir, '{}_{}_pipeline.pkl'.format(dataset, features)))
    models.save_model(model_step.model, os.path.join(save_dir, '{}_{}_model.h5'.format(dataset, features)))
