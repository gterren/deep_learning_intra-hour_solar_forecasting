# Deep Learning for Intra-Houra Solar Forecasting

This repository contains the files implemented for the different experiments carried out to develop an optimal Deep Learning (DL) intra-hour solar forecasting using infrared ground-based images. The model is based on deep recurrent networks architectures. 

The utils necessary to run the codes are in solar_forecasting_utils_v3,py and feature_extraction_utils.py (posted in this other repository https://github.com/gterren/kernel_intra-hour_solar_forecasting).

The state-of-the-art is implemented in solarnet.py and sunset.py for comparission purposes.

## Multi-Ouput Regression

The investigation aims first to find an optimaml Multilayer Perceptron architecture (MLP) to perform a multi-output forecast. Later, different recurrent layers are cross-validated to find the optimal combination of feature vector and recurrent architecture. Lastly, the best archicture is implemented and the its structural hyperparameters are cross-validated using variational inference.

### Information Fusion

The objective is to analyzed data adquired from different sensors: solar tracker, pyranometer, sky imager and weather station. In addition, the combination of multiple cloud features and its performances are also analyzed. The different horizons in the forecast are used to derive the Sun intersecting probability of each pixel in the image. Each of the distribution are considered a source of cloud dyanamics features and used in recurrent architecture.

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
