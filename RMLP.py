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
from solar_forecasting_utils_v2 import *
from bayesian_optimization_utils import *

# Do not display warnings in the output file
warnings.filterwarnings('ignore')

# Multi-Layer Perceptron
def _MLP(n_inputs, n_outputs, hidden_units_, dropout):
    # Build a simple model
    _inputs = keras.Input(shape = (n_inputs))
    _hidden = _inputs
    # Create hidden layers with deterministic weights using the Dense layer.
    for units in hidden_units_:
        _hidden = layers.Dense(units, activation = 'relu')(_hidden)
        _hidden = layers.Dropout(dropout)(_hidden)
    _outputs = layers.Dense(units = n_outputs)(_hidden)
    _outputs = layers.Dropout(dropout)(_outputs)
    # The output is deterministic: a single point estimate.
    return keras.Model(inputs = _inputs, outputs = _outputs)

# Define Architecture given the number of layers and Initial no. hidden neuros
def _define_architecture(N_layers, N_neurons):
    hidden_units_ = []
    for l in range(1, N_layers):
        hidden_units_.append(int(N_neurons/l))
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

# Train Recursive MLP models
def _model_training(X_tr_, Y_tr_, X_ts_, Y_ts_, theta_, i_lay, path):
    print(i_lay, np.exp(theta_[0]), int(np.exp(theta_[1])), np.exp(theta_[2]), np.exp(theta_[3]))
    model_name = path + r'RMLP_v31-1_{}{}.h5'.format(i_cov, i_lay)
    # Perform Neural Network Prediction
    n_inputs  = X_tr_.shape[1]
    n_outputs = Y_tr_.shape[1]
    n_samples = X_tr_.shape[0]
    # Get Datasets
    training_data_, validation_data_, testing_data_ = _get_training_validation_testing_dataset(X_tr_, Y_tr_, X_ts_, Y_ts_)
    # Get Predictors and Covariates for each partition of the dataset
    X_tr_prime_, Y_tr_   = training_data_
    X_val_prime_, Y_val_ = validation_data_
    X_ts_prime_, Y_ts_   = testing_data_
    # Define Architecture to Validate
    hidden_units_ = _define_architecture(N_layers  = i_lay,
                                         N_neurons = np.exp(theta_[2]))
    # Defime MLP model
    t_tr   = time.time()
    _model = _MLP(n_inputs, n_outputs, hidden_units_, dropout = np.exp(theta_[3]))
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
    return mean_absolute_percentage_error(Y_ts_[:, 0], Y_hat_ts_[:, 0]), Y_hat_ts_, [t_tr, t_ts]

# Validate Parameters set using Kfold cross-validation
def _kfold_cross_validation(theta_, args_):
    # Unpack Dataset and Constants
    X_tr_, Y_tr_, i_lay, path = args_
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
        error = _model_training(X_val_tr_, Y_val_tr_, X_val_ts_, Y_val_ts_, theta_, i_lay, path)[0]
        error_.append(error)
    return np.mean(error_)

# BORMLP K-Fold Cross-Validation of the model Parameters
def _get_BORMLP_cross_validation(X_tr_, Y_tr_, i_lay, path):
    # Define MLP parameters to validate
    learning_rate_ = (-6., -2.)
    batch_size_    = ( 5.,  9.)
    N_neurons_     = ( 1.,  5.)
    dropout_       = (-10,  0.)
    bounds_ = _BO_bounds(theta_ = [learning_rate_, batch_size_, N_neurons_, dropout_])
    # Constants Initialization
    args_     = (X_tr_, Y_tr_, i_lay, path)
    return _BO(_kfold_cross_validation, bounds_ = bounds_, _aqf = 'EI', xi = 1., kappa = 10.,
               X_0_ = _random_init(_kfold_cross_validation, bounds_, args_, n_init = 40),
               n_iterations = 70, maximize = True, args_ = args_, n_restarts = 10, display = True)

def _get_covariates(i_task, i_cov = None, i_sec = None):
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [0, 1, 2, 3, 4, 5]
    # Dataset Covariantes and Predictors Definition
    if i_task == 'persistence': return [idx_pred, idx_pred_horizon_, [0], [], 0, 0, [], [0, 1, 2, 3, 4, 5], []]
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [0, 1, 2, 3, 4, 5]
    # All
    if (i_task == 0) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 1) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 2) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 3) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 4) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    # Neibors order 1
    if (i_task == 0) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3]
    if (i_task == 1) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3, 4]
    if (i_task == 2) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 3) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 4) and (i_sec == 1): idx_cov_horizon_  = [1, 2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 1): idx_cov_horizon_  = [2, 3, 4, 5]
    # Neibors order 2
    if (i_task == 0) and (i_sec == 2): idx_cov_horizon_  = [0, 1, 2]
    if (i_task == 1) and (i_sec == 2): idx_cov_horizon_  = [0, 1, 2, 3]
    if (i_task == 2) and (i_sec == 2): idx_cov_horizon_  = [1, 2, 3, 4]
    if (i_task == 3) and (i_sec == 2): idx_cov_horizon_  = [1, 2, 3, 4]
    if (i_task == 4) and (i_sec == 2): idx_cov_horizon_  = [2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 2): idx_cov_horizon_  = [3, 4, 5]
    # Neibors order 3
    if (i_task == 0) and (i_sec == 3): idx_cov_horizon_  = [0, 1]
    if (i_task == 1) and (i_sec == 3): idx_cov_horizon_  = [0, 1, 2]
    if (i_task == 2) and (i_sec == 3): idx_cov_horizon_  = [1, 2, 3]
    if (i_task == 3) and (i_sec == 3): idx_cov_horizon_  = [2, 3, 4]
    if (i_task == 4) and (i_sec == 3): idx_cov_horizon_  = [3, 4, 5]
    if (i_task == 5) and (i_sec == 3): idx_cov_horizon_  = [4, 5]
    # Neibors order 4
    if (i_task == 0) and (i_sec == 4): idx_cov_horizon_  = [0, 1, 2, 3]
    if (i_task == 1) and (i_sec == 4): idx_cov_horizon_  = [0, 1, 2]
    if (i_task == 2) and (i_sec == 4): idx_cov_horizon_  = [1, 2, 3]
    if (i_task == 3) and (i_sec == 4): idx_cov_horizon_  = [2, 3, 4]
    if (i_task == 4) and (i_sec == 4): idx_cov_horizon_  = [2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 4): idx_cov_horizon_  = [4, 5]
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
# Generate Persistence
X_, Y_, Z_ = _generate_database(dataset_, cov_idx_ = _get_covariates('persistence'))
y_tr_hat_persistence_, Y_tr_, _, y_ts_hat_persistence_, Y_ts_, _ = _split_dataset(X_, Y_, Z_, percentage = 0.8)
print(Y_tr_.shape, y_tr_hat_persistence_.shape, Y_ts_.shape, y_ts_hat_persistence_.shape)
# Make a Persistent Prediction and evaluate error
e_ts_persistence_ = mean_absolute_percentage_error(Y_ts_, y_ts_hat_persistence_)
print(e_ts_persistence_, e_ts_persistence_.mean())

if validation:
    # Nodes and jobs information for communication from MPI
    i_cov = int(sys.argv[1])
    i_lay = int(sys.argv[2])
    i_sec = 4
    # Get Experiment for the i-th Job
    print(i_cov, i_lay, i_sec)

    Theta_     = []
    error_val_ = []
    error_ts_  = []
    time_      = []
    Y_ts_hat_  = np.empty((Y_ts_.shape[0], 0))
    # Loop Over Horizons
    for i_task in range(Y_tr_.shape[1]):
        print(i_task + 1, Y_tr_.shape[1])
        # Generate database
        X_, Y_, Z_ = _generate_database(dataset_, cov_idx_ = _get_covariates(i_task, i_cov, i_sec))
        X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, percentage = 0.8)
        print(X_tr_.shape, Y_tr_.shape, Z_tr_.shape, X_ts_.shape, Y_ts_.shape, Z_ts_.shape)
        # Recurrent Prediction Concatenating
        X_tr_ = np.concatenate((X_tr_, Y_tr_[:, :i_task]), axis = 1)
        X_ts_ = np.concatenate((X_ts_, Y_ts_hat_), axis = 1)
        # Find Optimal Recurrent MLPs Architecture Parameters
        theta_, error_val = _get_BORMLP_cross_validation(X_tr_, Y_tr_[:, i_task][:, np.newaxis], i_lay, path = r'/users/terren/solar_forecasting/model/deep_learning/')
        print(theta_, -error_val)
        # Training Optimal Recurrent MLPs Architecture Parameters
        tm1                 = time.time()
        error_ts, y_ts_hat_ = _model_training(X_tr_, Y_tr_[:, i_task][:, np.newaxis],
                                              X_ts_, Y_ts_[:, i_task][:, np.newaxis],
                                              theta_, i_lay, path = r'/users/terren/solar_forecasting/model/deep_learning/')[0, 1]
        tm                  = time.time() - tm1
        print(tm, error_ts)
        Y_ts_hat_ = np.concatenate((Y_ts_hat_, y_ts_hat_), axis = 1)
        # Save Results for current Horizon
        Theta_.append(theta_)
        error_val_.append(-error_val)
        error_ts_.append(error_ts)
        time_.append(tm)

    # Save Results by row in a .csv file
    Theta_     = np.stack(Theta_).flatten()
    error_val_ = np.stack(error_val_).flatten()
    x_ = [[i_cov, i_lay, i_sec]] + [Theta_.tolist()] + error_val_.tolist() + error_ts_ + time_ + e_ts_persistence_.tolist()
    print(x_)
    # Dump data in a .csv
    name = r'/users/terren/solar_forecasting/logs/neural_networks/RMLP_v31-1.csv'
    with open(name, 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(x_)

else:
    i_init = int(sys.argv[1])
    # 0.15818217844282878 330.5683410167694 [7. 2. 4.] [-5.950718821762995, 6.185752295736515, 2.5227524996898394, -9.033858980630775, -4.430108489423077, 8.434728234663124, 1.5476663782791107, -8.192006426551378, -5.739064221631313, 7.45679015547417, 1.9828612495690083, -9.978735736249442, -5.592637759385827, 8.742786686282024, 1.011332742484282, -9.406923567248246, -5.38413063100533, 8.837387515420932, 1.7728559411885643, -8.943403380977294, -5.476668994332172, 8.954155087881357, 1.7244887692706032, -9.922089309722946]
    Theta_  = [[-5.950718821762995, 6.185752295736515, 2.5227524996898394, -9.033858980630775],
               [-4.430108489423077, 8.434728234663124, 1.5476663782791107, -8.192006426551378],
               [-5.739064221631313,  7.45679015547417, 1.9828612495690083, -9.978735736249442],
               [-5.592637759385827, 8.742786686282024,  1.011332742484282, -9.406923567248246],
               [ -5.38413063100533, 8.837387515420932, 1.7728559411885643, -8.943403380977294],
               [-5.476668994332172, 8.954155087881357, 1.7244887692706032, -9.922089309722946]]
    i_cov   = 7
    i_lay   = 2
    i_sec   = 4
    # Get Experiment for the i-th Job
    print(i_cov, i_lay, i_sec)

    error_ts_ = []
    Y_ts_hat_ = np.empty((Y_ts_.shape[0], 0))
    t_tr      = 0
    t_ts      = 0
    # Loop Over Horizons
    for i_task in range(Y_tr_.shape[1]):
        print(i_task + 1, Y_tr_.shape[1])
        # Generate database
        X_, Y_, Z_ = _generate_database(dataset_, cov_idx_ = _get_covariates(i_task, i_cov, i_sec))
        X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, percentage = 0.8)
        print(X_tr_.shape, Y_tr_.shape, Z_tr_.shape, X_ts_.shape, Y_ts_.shape, Z_ts_.shape)
        # Recurrent Prediction Concatenating
        X_tr_ = np.concatenate((X_tr_, Y_tr_[:, :i_task]), axis = 1)
        X_ts_ = np.concatenate((X_ts_, Y_ts_hat_), axis = 1)
        # Training Optimal Recurrent MLPs Architecture Parameters
        error_ts, y_ts_hat_, time_ = _model_training(X_tr_, Y_tr_[:, i_task][:, np.newaxis],
                                                     X_ts_, Y_ts_[:, i_task][:, np.newaxis],
                                                     Theta_[i_task], i_lay, path = r'/users/terren/solar_forecasting/model/deep_learning/test_{}_'.format(i_init))
        Y_ts_hat_ = np.concatenate((Y_ts_hat_, y_ts_hat_), axis = 1)
        # Save Results for current Horizon
        error_ts_.append(error_ts)
        t_tr += time_[0]
        t_ts += time_[1]

    print(error_ts_, np.mean(error_ts_), t_tr, t_ts)

    np.savetxt(r'/users/terren/solar_forecasting/model/deep_learning/test_{}_RMLP_prediction.csv'.format(i_init), Y_ts_hat_, delimiter = ',')

    x_ = [i_init, time_[0], time_[1], np.mean(error_ts_)] + error_ts_

    name = r'/users/terren/solar_forecasting/logs/neural_networks/test_RMLP_v31-1.csv'
    with open(name, 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(x_)
