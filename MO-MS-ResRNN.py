import pickle, glob, sys, os, warnings, csv
import numpy as np

from datetime import datetime
import time

from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from scipy.stats import multivariate_normal

import scipy.special as ss
from scipy import optimize

from feature_extraction_utils import _load_file, _save_file
from solar_forecasting_utils_v3 import *
from bayesian_optimization_utils import *

# Do not display warnings in the output file
warnings.filterwarnings('ignore')

# Divide Feature Vectors per Source of FeAtures
def _split_features(structure, tags_):
    # Remove Empty Index before returing them
    def __idx_return(idx_):
        idx_return_ = []
        for idx in idx_:
            if len(idx) != 0: idx_return_.append(idx)
        return idx_return_
    # Get Features in the Selection
    feature_tags_ = []
    for tag_ in tags_:
        feature_tags_.append(tag_[:1])
    # Labels to Index
    feature_tags_ = np.stack(feature_tags_)
    idx_ = np.arange(len(feature_tags_), dtype = int)
    # Group Features by Source
    i_idx_, a_idx_, t_idx_, h_idx_, m_idx_, d_idx_, v_idx_ = [], [], [], [], [], [], []
    for tag in np.unique(feature_tags_):
        if tag == 'i': i_idx_ = idx_[feature_tags_ == tag].tolist()
        if tag == 'a': a_idx_ = idx_[feature_tags_ == tag].tolist()
        if tag == 't': t_idx_ = idx_[feature_tags_ == tag].tolist()
        if tag == 'h': h_idx_ = idx_[feature_tags_ == tag].tolist()
        if tag == 'm': m_idx_ = idx_[feature_tags_ == tag].tolist()
        if tag == 'd': d_idx_ = idx_[feature_tags_ == tag].tolist()
        if tag == 'v': v_idx_ = idx_[feature_tags_ == tag].tolist()
    # Get Index of all MA features
    f_idx_ = np.asarray(np.concatenate((a_idx_, t_idx_, h_idx_, m_idx_, d_idx_, v_idx_), axis = 0), dtype = int)
    # Return AR and MA dataset
    if i_structure == 0:
        return __idx_return([i_idx_, f_idx_])
    # Return Sources Dataset
    if i_structure == 1:
        return __idx_return([i_idx_, a_idx_, t_idx_, h_idx_, m_idx_, d_idx_, v_idx_])
    # Get Index of all Sector features
    f_idx_ = np.asarray(np.concatenate((t_idx_, h_idx_, m_idx_, d_idx_, v_idx_), axis = 0), dtype = int)
    # Group Features by Sector
    s0_idx_, s1_idx_, s2_idx_, s3_idx_, s4_idx_, s5_idx_ = [], [], [], [], [], []
    for f_idx, tag_ in zip(f_idx_.tolist(), tags_[f_idx_.tolist()]):
        if tag_[-5:-3] == 's0': s0_idx_.append(f_idx)
        if tag_[-5:-3] == 's1': s1_idx_.append(f_idx)
        if tag_[-5:-3] == 's2': s2_idx_.append(f_idx)
        if tag_[-5:-3] == 's3': s3_idx_.append(f_idx)
        if tag_[-5:-3] == 's4': s4_idx_.append(f_idx)
        if tag_[-5:-3] == 's5': s5_idx_.append(f_idx)
    # Return Sectors Dataset
    if structure == 2:
        return __idx_return([i_idx_, a_idx_, s0_idx_, s1_idx_, s2_idx_, s3_idx_, s4_idx_, s5_idx_])

# Enconder-Decoder Recurrent NN paralled with MLP
def _MLP_RNN(n_inputs, n_outputs, R_layers_i, R_layers_f, rnn_hidden_units_i_, rnn_hidden_units_f_, mlp_hidden_units_, dropout):
    # Split the Feature Vector in different Inputs
    def __feature_vector(x_, idx_):
        return tf.gather(x_, idx_, axis = 1)
    # Define the Input Layer
    def __input_layer(_hidden, rnn_hidden_units_, R_layer, dropout):
        # Create hidden layers using recurrent layer.
        N_R_lay = len(rnn_hidden_units_)
        for units, i in zip(rnn_hidden_units_, range(N_R_lay)):
            # Return Sequences Only when multiple RNN are in serie
            if (N_R_lay == 2) and (i == 0):
                return_sequences = True
            else:
                return_sequences = False
            # Recurrent Layer
            if R_layer == 0:
                # Reshape Output for Encoder-Decoder
                _hidden = layers.SimpleRNN(units, activation        = 'tanh',
                                                  dropout           = dropout,
                                                  return_sequences  = return_sequences)(_hidden)
            # Long-Short Term Memory Layer
            if R_layer == 1:
                # Reshape Output for Encoder-Decoder
                _hidden = layers.LSTM(units, activation           = 'tanh',
                                             recurrent_activation = 'sigmoid',
                                             dropout              = dropout,
                                             return_sequences     = return_sequences)(_hidden)
            # Gated Recurrent Layer
            if R_layer == 2:
                _hidden = layers.GRU(units, activation           = 'tanh',
                                            recurrent_activation = 'sigmoid',
                                            dropout              = dropout,
                                            return_sequences     = return_sequences)(_hidden)
        return _hidden
    # Residual Layers
    def _residuals_layer(_hidden, _output, dropout):
        print(_hidden.shape, _output.shape)
        _hidden = layers.Flatten()(_hidden)
        _hidden = layers.Dense(_output.shape[-1], activation = 'relu')(_hidden)
        _hidden = layers.Dropout(dropout)(_hidden)
        _hidden = layers.BatchNormalization()(_hidden)
        print(_hidden.shape, _output.shape)
        return tf.keras.layers.Add()([_output, _hidden])
    # Feature Reshape for Rucurrent Layers
    def _get_features_input(_inputs, forward):
        _hidden_2 = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[2]}))(_inputs)
        _hidden_2 = layers.Reshape((1, _hidden_2.shape[1]))(_hidden_2)
        _hidden_3 = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[3]}))(_inputs)
        _hidden_3 = layers.Reshape((1, _hidden_3.shape[1]))(_hidden_3)
        _hidden_4 = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[4]}))(_inputs)
        _hidden_4 = layers.Reshape((1, _hidden_4.shape[1]))(_hidden_4)
        _hidden_5 = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[5]}))(_inputs)
        _hidden_5 = layers.Reshape((1, _hidden_5.shape[1]))(_hidden_5)
        _hidden_6 = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[6]}))(_inputs)
        _hidden_6 = layers.Reshape((1, _hidden_6.shape[1]))(_hidden_6)
        _hidden_7 = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[7]}))(_inputs)
        _hidden_7 = layers.Reshape((1, _hidden_7.shape[1]))(_hidden_7)
        if forward: return keras.layers.concatenate([_hidden_7, _hidden_6, _hidden_5, _hidden_4, _hidden_3, _hidden_2], axis = 1)
        else:       return keras.layers.concatenate([_hidden_2, _hidden_3, _hidden_4, _hidden_5, _hidden_6, _hidden_7], axis = 1)
    # Defien inputs
    _inputs = keras.Input(shape = (n_inputs))
    # Define The Input Layers for the autoregresive source
    _hidden_i = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[0]}))(_inputs)
    _hidden_i = layers.Reshape((_hidden_i.shape[1], 1))(_hidden_i)
    _hidden_a = keras.layers.Lambda(__feature_vector, arguments = ({'idx_': index_[1]}))(_inputs)
    _hidden_f_forward = _get_features_input(_inputs, forward = True)
    # Intput Recurrent Layers
    _output_i         = __input_layer(_hidden_i, rnn_hidden_units_i_, R_layers_i, dropout)
    _output_f_forward = __input_layer(_hidden_f_forward, rnn_hidden_units_f_, R_layers_f, dropout)
    print(_output_i.shape, _hidden_a.shape, _output_f_forward.shape)
    # Intput Residual Layers
    _output_i          = _residuals_layer(_hidden_i, _output_i, dropout)
    _output_f_forward  = _residuals_layer(_hidden_f_forward, _output_f_forward, dropout)
    # Define The Input Layers for each feature source
    _hidden = keras.layers.concatenate([_output_i, _hidden_a, _output_f_forward], axis = 1)
    # Create hidden layers with deterministic weights using the Dense layer.
    for units in mlp_hidden_units_:
        _hidden = layers.Dense(units, activation = 'relu')(_hidden)
        _hidden = layers.Dropout(dropout)(_hidden)
    _outputs = layers.Dense(units = n_outputs)(_hidden)
    _outputs = layers.Dropout(dropout)(_outputs)
    # The output is deterministic: a single point estimate.
    return keras.Model(inputs = _inputs, outputs = _outputs)

# Define Architecture given the number of layers and Initial no. hidden neuros
def _define_rnn_architecture(R_layers, R_neurons):
    hidden_units_ = [int(R_neurons)]
    for l in range(2, R_layers + 1):
        hidden_units_.append(int(R_neurons/l))
    return hidden_units_

# Define Architecture given the number of layers and Initial no. hidden neuros
def _define_mlp_architecture(M_layers, M_neurons):
    hidden_units_ = []
    for l in range(1, M_layers):
        hidden_units_.append(int(M_neurons/l))
    return hidden_units_

# Defining Validation Dataset when traning for testing or validation
def _get_training_validation_testing_dataset(X_tr_, Y_tr_, X_ts_, Y_ts_, val_percentage = 0.075):
    # Data Normalization
    N_tr  = X_tr_.shape[0]
    N_ts  = X_ts_.shape[0]
    N_val = int(N_tr*val_percentage)
    #print(N_tr, N_val, N_ts)
    _scaler_x        = MinMaxScaler().fit(X_tr_[:-N_val, :])
    X_tr_prime_      = _scaler_x.transform(X_tr_[:-N_val, :])
    X_val_prime_     = _scaler_x.transform(X_tr_[-N_val:, :])
    X_ts_prime_      = _scaler_x.transform(X_ts_)
    # Define Testing Partitions
    training_data_   = (X_tr_prime_, Y_tr_[:-N_val, :])
    validation_data_ = (X_val_prime_, Y_tr_[-N_val:, :])
    testing_data_    = (X_ts_prime_, Y_ts_)
    return training_data_, validation_data_, testing_data_

# Train Independent MLP models
def _model_training(X_tr_, Y_tr_, X_ts_, Y_ts_, theta_, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, i_R_neurons, path):
    print(i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, np.exp(theta_[0]), int(np.exp(theta_[1])), np.exp(theta_[2]), np.exp(theta_[3]), np.exp(theta_[4]))
    model_name = path + r'MO-MS-ResRNN_v31-1_{}{}{}{}{}{}.h5'.format(i_cov, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f)
    # Perform Neural Network Prediction
    n_inputs  = X_tr_.shape[1]
    n_outputs = Y_tr_.shape[1]
    # Get Datasets
    training_data_, validation_data_, testing_data_ = _get_training_validation_testing_dataset(X_tr_, Y_tr_, X_ts_, Y_ts_)
    # Get Predictors and Covariates for each partition of the dataset
    X_tr_prime_, Y_tr_   = training_data_
    X_val_prime_, Y_val_ = validation_data_
    X_ts_prime_, Y_ts_   = testing_data_
    # Define Recurrent Architecture to Validate
    rnn_hidden_units_i_ = _define_rnn_architecture(R_layers = i_R_lay, R_neurons = np.exp(i_R_neurons))
    print(rnn_hidden_units_i_)
    rnn_hidden_units_f_ = _define_rnn_architecture(R_layers = f_R_lay, R_neurons = np.exp(theta_[2]))
    print(rnn_hidden_units_f_)
    # Define Dense Architecture to Validate
    mlp_hidden_units_ = _define_mlp_architecture(M_layers = i_M_lay, M_neurons = np.exp(theta_[3]))
    print(mlp_hidden_units_)
    # Defime RNN model
    t_tr   = time.time()
    _model = _MLP_RNN(n_inputs, n_outputs, R_lay_i, R_lay_f, rnn_hidden_units_i_, rnn_hidden_units_f_, mlp_hidden_units_, dropout = np.exp(theta_[4]))
    # Compile Model
    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = np.exp(theta_[0])),
                   loss      = keras.losses.MeanAbsolutePercentageError(),
                   metrics   = [keras.metrics.MeanAbsolutePercentageError()])
    # Train Neural Network
    _history = _model.fit(X_tr_prime_, Y_tr_, validation_data = (X_val_prime_, Y_val_),
                                              batch_size      = int(np.exp(theta_[1])),
                                              epochs          = 10000,
                                              verbose         = 0,
                                              callbacks       = [EarlyStopping(monitor  = 'val_loss',
                                                                               mode     = 'min',
                                                                               patience = 450,
                                                                               verbose  = 0),
                                                                 ModelCheckpoint(model_name,
                                                                                 monitor        = 'val_loss',
                                                                                 mode           = 'min',
                                                                                 verbose        = 0,
                                                                                 save_best_only = True)])
    t_tr   = time.time() - t_tr
    _model = load_model(model_name)
    # Compute validation loss and Error
    t_ts      = time.time()
    Y_hat_ts_ = _model.predict(X_ts_prime_, verbose = 0)
    t_ts      = time.time() - t_ts
    return mean_absolute_percentage_error(Y_ts_, Y_hat_ts_), Y_hat_ts_, [t_tr, t_ts]

# Validate Parameters set using Kfold cross-validation
def _kfold_cross_validation(theta_, args_):
    print(theta_)
    # Unpack Dataset and Constants
    X_tr_, Y_tr_, R_layers_i, R_layers_f, M_layers, R_layer_i, R_lay_f, apth = args_
    # Define Storage Variables
    error_ = []
    # Loop Over K-folds
    for idx_tr_, idx_ts_ in KFold(n_splits     = 3,
                                  random_state = None,
                                  shuffle      = False).split(X_tr_):
        # Split Validation Set in training and test
        X_val_tr_, X_val_ts_ = X_tr_[idx_tr_, :], X_tr_[idx_ts_, :]
        Y_val_tr_, Y_val_ts_ = Y_tr_[idx_tr_, :], Y_tr_[idx_ts_, :]
        # Traning Model
        error = _model_training(X_val_tr_, Y_val_tr_, X_val_ts_, Y_val_ts_, theta_,
                                R_layers_i, R_layers_f, M_layers, R_layer_i, R_lay_f, i_R_neurons = 3.5036535952058725, path = path)[0].mean()
        error_.append(error)
    return np.mean(error_)

# BO-MO-MLP-RNN K-Fold Cross-Validation of the model Parameters
def _get_BO_MO_MLP_RNN_cross_validation(X_tr_, Y_tr_, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, apth):
    # Define RNN parameters to validate
    learning_rate_ = (-6.,-2.)
    batch_size_    = ( 5.,7.5)
    f_R_neurons_   = ( 1.,4.5)
    M_neurons_     = (1.5, 5.)
    dropout_       = (-10,-2.)
    bounds_ = _BO_bounds(theta_ = [learning_rate_, batch_size_, f_R_neurons_, M_neurons_, dropout_])
    # Constants Initialization
    args_     = (X_tr_, Y_tr_, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, path)
    return _BO(_kfold_cross_validation, bounds_      = bounds_,
                                        _aqf         = 'EI',
                                        xi           = 1.,
                                        kappa        = 10.,
                                        X_0_         = _random_init(_kfold_cross_validation, bounds_, args_, n_init = 35),
                                        n_iterations = 60,
                                        maximize     = True,
                                        args_        = args_,
                                        n_restarts   = 10,
                                        display      = True)

def _get_covariates(i_cov):
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [0, 1, 2, 3, 4, 5]
    # Dataset Covariantes and Predictors Definition
    if i_cov == 'persistence': return [idx_pred, idx_pred_horizon_, [0], [], 0, 0, [], [0, 1, 2, 3, 4, 5], []]
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    # Cross-validation of CSI AR
    cov_idx_0_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [], 0, 0, [], idx_cov_horizon_, []]
    # Cross-validation of CSI AR + Angles
    cov_idx_1_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 0, [], idx_cov_horizon_, []]
    # Cross-validation of CSI AR + Angles + Raw Temperatures
    cov_idx_2_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 0, [0, 1], idx_cov_horizon_, [0]]
    # Cross-validation of CSI AR + Angles + Processed Temperatures
    cov_idx_3_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 3, 0, [0, 1], idx_cov_horizon_, [0]]
    # Cross-validation of CSI AR + Angles + Processed Heights
    cov_idx_4_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [1]]
    # Cross-validation of CSI AR + Angles + Raw Temperatures + Processed Heights
    cov_idx_5_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1]]
    # Cross-validation of CSI AR + Angles + Raw Temperatures + Processed Heights + Magnitude
    cov_idx_6_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1, 2]]
    # Cross-validation of CSI AR + Angles + Raw Temperatures + Processed Heights + Magnitude + Divergence
    cov_idx_7_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1, 2, 4]]
    # Cross-validation of CSI AR + Angles + Raw Temperatures + Processed Heights + Magnitude + Divergence + Vorticity
    cov_idx_8_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1, 2, 3, 4]]
    # Index of all Covariances
    return [cov_idx_0_,  cov_idx_1_,  cov_idx_2_,  cov_idx_3_,  cov_idx_4_,  cov_idx_5_, cov_idx_6_, cov_idx_7_, cov_idx_8_][i_cov]

validation = False

name = r'/xena/scratch/terren/database_feature_selection_v31-1/*'
print(name)

# Load Dataset
dataset_ = _load_dataset(name)
print(len(dataset_))
# Generate database
X_, Y_, Z_ = _generate_database(dataset_, cov_idx_ = _get_covariates('persistence'))
y_tr_hat_persistence_, Y_tr_, _, y_ts_hat_persistence_, Y_ts_, _ = _split_dataset(X_, Y_, Z_, percentage = 0.8)
print(Y_tr_.shape, y_tr_hat_persistence_.shape, Y_ts_.shape, y_ts_hat_persistence_.shape)
# Make a Persistent Prediction and evaluate error
e_ts_persistence_ = mean_absolute_percentage_error(Y_ts_, y_ts_hat_persistence_)
print(e_ts_persistence_, e_ts_persistence_.mean())

if validation:
    # Nodes and jobs information for communication from MPI
    i_cov   = int(sys.argv[1])
    i_R_lay = 2
    f_R_lay = int(sys.argv[2])
    i_M_lay = int(sys.argv[3])
    R_lay_i = 0
    R_lay_f = int(sys.argv[4])
    i_structure = 2
    # Get Experiment for the i-th Job
    #6 2 1 3 0 3 2
    print(i_cov, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, i_structure)

    cov_idx_ = _get_covariates(i_cov)
    # Get Index of The Dataset
    index_ = _split_features(structure = i_structure, tags_ =_get_sample_tags(cov_idx_))
    print(len(index_))
    # Generate database
    X_, Y_, Z_ = _generate_database(dataset_, cov_idx_)
    X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, percentage = 0.8)
    print(X_tr_.shape, Y_tr_.shape, Z_tr_.shape, X_ts_.shape, Y_ts_.shape, Z_ts_.shape)
    # Find Optimal MO-MLP-RNNs Architecture Parameters
    theta_, error_val_ = _get_BO_MO_MLP_RNN_cross_validation(X_tr_, Y_tr_, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, path = r'/users/terren/solar_forecasting/model/deep_learning/')
    print(theta_, -error_val_)
    # Training Optimal MO-MLP-RNNS Architecture Parameters
    tm1       = time.time()
    error_ts_ = _model_training(X_tr_, Y_tr_, X_ts_, Y_ts_, theta_, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, i_R_neurons = 3.5036535952058725,
                                                                                                                 path        = r'/users/terren/solar_forecasting/model/deep_learning/')[0]
    tm        = time.time() - tm1
    print(tm, error_ts_, error_ts_.mean())
    # Save Results by row in a .csv file
    x_ = [[i_cov, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f]] + [np.stack(theta_).tolist()] + np.stack(error_val_).tolist() + np.stack(error_ts_).tolist() + [tm] + e_ts_persistence_.tolist()
    # Dump data in a .csv
    name = r'/users/terren/solar_forecasting/logs/neural_networks/MO-MS-ResRNN_v31-1.csv'
    with open(name, 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(x_)

else:
    i_init = int(sys.argv[1])
    #0.15411066821793426 431.1413426399231 [6. 2. 2. 2. 0. 1.] [-5.817247859918967, 5.462991650051459, 1.7967635536615394, 2.2223169437735706, -9.465462043186191]
    theta_  = [-5.817247859918967, 5.462991650051459, 1.7967635536615394, 2.2223169437735706, -9.465462043186191]
    i_cov   = 6
    i_R_lay = 2
    f_R_lay = 2
    i_M_lay = 2
    R_lay_i = 0
    R_lay_f = 1
    i_structure = 2
    # Get Experiment for the i-th Job
    #6 2 1 3 0 3 2
    print(i_cov, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, i_structure)

    cov_idx_ = _get_covariates(i_cov)
    # Get Index of The Dataset
    index_ = _split_features(structure = i_structure, tags_ =_get_sample_tags(cov_idx_))
    print(len(index_))
    # Generate database
    X_, Y_, Z_ = _generate_database(dataset_, cov_idx_)
    X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, percentage = 0.8)
    print(X_tr_.shape, Y_tr_.shape, Z_tr_.shape, X_ts_.shape, Y_ts_.shape, Z_ts_.shape)
    # Training Optimal MO-MLP-RNNS Architecture Parameters
    error_ts_, Y_ts_hat_, time_ = _model_training(X_tr_, Y_tr_, X_ts_, Y_ts_, theta_, i_R_lay, f_R_lay, i_M_lay, R_lay_i, R_lay_f, i_R_neurons = 3.5036535952058725,
                                                                                                                                   path        = r'/users/terren/solar_forecasting/model/deep_learning/test_{}_'.format(i_init))

    print(error_ts_, np.mean(error_ts_), time_)

    np.savetxt(r'/users/terren/solar_forecasting/model/deep_learning/test_{}_MO-MS-ResRNN_prediction.csv'.format(i_init), Y_ts_hat_, delimiter = ',')

    x_ = [i_init, time_[0], time_[1], np.mean(error_ts_)] + error_ts_.tolist()

    name = r'/users/terren/solar_forecasting/logs/neural_networks/test_MO-MS-ResRNN_v31-1.csv'
    with open(name, 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(x_)
