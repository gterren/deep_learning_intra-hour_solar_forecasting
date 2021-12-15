# Deep Learning for Intra-Houra Solar Forecasting



## Multi-Ouput Regression

### Independent Architecture

The multi-output MLP achitecture composed of independet MLPs in parallel is in MLP.py

### Recursive Architecture

The multi-output MLP achitecture form of multiple recursive MLPs is in RMLP.py

### Multi-Task Architecture

The multi-output MLP achitecture that uses a single multi-taks MLP is MO-MLP.py

## Recurrent Networks

The type of Recurrent Neural Networks (RNNs) cross-validated for each DL architecture are: Simple Recurrent, Long-Short Term Memory, and Gated Recurrent Units.

## Bidirectional Architecture

The bidirecional architecture applied to the RNNs in MO-MS-BiRNN.py and MO-MS-ResBiRNN.py (without and with residual layer respectively).

## Residudal Networks

The RNNs architecture with residual layers are MO-ResRNN.py (AR model), MO-MS-ResRNN.py (AR-Multiple Source model) and MO-MS-ResBiRNN.py (AR-Multiple Source with Bidirectional architecture in the multiple sources model). The bayesian implementation of MO-MS-ResRNN.py is in MO-MS-BayResRNN.py.

## Probabilistic Networks

The file contaning the implementation of a varational MLP for a deep RNNs is MO-MS-BayResRNN.py

## Bayesian Optimization

The library of Bayesian Optimization to implemente an efficient cross-validation of the strucural parameters is in bayesian_optimization_utils.py. 
