import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel


def _BO_bounds(theta_):
    bounds_ = []
    for theta in theta_:
        if theta != 0.:
            bounds_.append((theta[0], theta[1]))
    return tuple(bounds_)


def _random_init(_f, bounds_, args_ = (), n_init = 1):
    X_0_ = np.zeros((n_init, len(bounds_)))
    y_0_ = np.zeros((n_init))

    for i in range(len(bounds_)):
        X_0_[:, i] = np.random.uniform(bounds_[i][0], bounds_[i][1], n_init)

    for i in range(n_init):
        y_0_[i] = _f(X_0_[i, :], args_)
    return np.concatenate((X_0_, y_0_[:, np.newaxis]), axis = 1)



def _BO(_f, bounds_, _aqf = None, xi = None, kappa = None, X_0_ = None, kernel = r'matern',
                           n_iterations = None, maximize = True, args_ = (), n_restarts = 25, display = True):

    # AQF: Evaluate Upper Confidence Bound Function
    def __UCB(x_,  y_, xi, kappa, _GP):
        ''' Computes the UCB at points X based on existing samples
        X_ and y_ using a Gaussian process for regression.
        Args:
            x:     Points at which UCB shall be computed (m x d).
            X_:    Sample locations (n x d). Y_sample: Sample values (n x 1).
            _GP:   A GaussianProcessRegressor fitted to samples.
            kappa: Exploitation-exploration trade-off parameter
        Returns:
            Upper Confident Interval at pointx x. '''
        # Evaluate Gaussian Process
        mu, s = _GP.predict(x_[np.newaxis], return_std = True)
        # Evaluate UCB function
        ucb   = mu + kappa*s
        return - ucb[0]
    # AQF: Evaluate Expected Improvement Function
    def __EI(x_, y_, xi, kappa, _GP):
        ''' Computes the EI at points X based on existing samples
        X_ and y_ using a Gaussian process for regression.
        Args:
            x:   Points at which EI shall be computed (m x d).
            X_:  Sample locations (n x d). Y_sample: Sample values (n x 1).
            _GP: A GaussianProcessRegressor fitted to samples.
            xi:  Exploitation-exploration trade-off parameter.
        Returns:
            Expected improvements at points x. '''
        # Evaluate Gaussian Process
        mu, s = _GP.predict(x_[np.newaxis], return_std = True)
        # Evaluate EI function
        y_max = y_.max()
        z  = (mu - y_max - xi) / s
        ei = (mu - y_max - xi) * norm.cdf(z) + s * norm.pdf(z)
        return -ei[0]

    # ACF: Evaluate Maximum Probability of Improvement Function
    def __MPI(x_, y_, xi, kappa, _GP):
        ''' Computes the MPI at points X based on existing samples
        X_ and y_ using a Gaussian process for regression.
        Args:
            x:   Points at which MPI shall be computed (m x d).
            X_:  Sample locations (n x d). Y_sample: Sample values (n x 1).
            _GP: A GaussianProcessRegressor fitted to samples.
            xi:  Exploitation-exploration trade-off parameter.
        Returns:
            Maximum Probability of Improvement at points x. '''
        # Evaluate Gaussian Process
        mu, s = _GP.predict(x_[np.newaxis], return_std = True)
        # Evaluate MPI function
        y_max = y_.max()
        z   = (mu - y_max - xi) / s
        mpi = norm.cdf(z)
        return -mpi[0]

    # Display Point Sample in Current Interation
    def __diplay(i, Y_k_, x_k_1_, y_k_1_, maximize):
        if maximize:
            # Display Marker if lowest Point yet
            if Y_k_.min() < y_k_1_: print(r'No. Iter. = {} X = {} Y = {}'.format(i, x_k_1_, -y_k_1_))
            else:                   print(r'No. Iter. = {} X = {} Y = {} <<--- '.format(i, x_k_1_, -y_k_1_))
        else:
            # Display Marker if lowest Point yet
            if Y_k_.min() < y_k_1_: print(r'No. Iter. = {} X = {} Y = {}'.format(i, x_k_1_, y_k_1_))
            else:                   print(r'No. Iter. = {} X = {} Y = {} <<--- '.format(i, x_k_1_, y_k_1_))
    # Run optimization of the adquisition function
    def __optimize(_aqf, _GP, X_k_, y_k_, bounds_, maximize):
        ''' Proposes the next sampling point by optimizing the acquisition function.
        Args:     acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        GP:       A GaussianProcessRegressor fitted to samples.
        Returns:  Location of the acquisition function maximum. '''

        # Initialize arguments for the adquisiton function
        args_ = (y_k_, xi, kappa, _GP)
        # Storage Vriable Initialization
        X_0_  = np.empty((n_restarts, 0))
        # Start optimization from n_restart different points.
        for bound_ in bounds_:
            X_0_ = np.concatenate((X_0_, np.random.uniform(bound_[0], bound_[1], size = (n_restarts, 1))), axis = 1)
        # Storage Variables
        X_opt_, y_opt_ = [], []
        # Loop over initialization
        for i in range(X_0_.shape[0]):
            _opt = minimize(_aqf, x0 = X_0_[i, :], bounds = bounds_, args = args_, method = 'L-BFGS-B')
            # Save Optimization Results
            X_opt_.append(_opt.x)
            y_opt_.append(_opt.fun[0])
        # Return the best optimal point
        return X_opt_[np.argmin(y_opt_)]

    # Define the adq function selected by the user
    if _aqf == r'EI':  _aqf = __EI
    if _aqf == r'MPI': _aqf = __MPI
    if _aqf == r'UCB': _aqf = __UCB
    # Gaussian process for regresison with Matern 5/2 kernel
    if kernel == r'rbf':    _k = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel()
    if kernel == r'matern': _k = ConstantKernel() * Matern(nu = 5./2.) + WhiteKernel() + ConstantKernel()
    _GP = GaussianProcessRegressor(kernel = _k, alpha = 1e-5, normalize_y = True, n_restarts_optimizer = 7)
    # Initiliaze Algorithm with samples supplyed by user
    if X_0_.shape[1] == 2:
        X_k_ = X_0_[..., 0]
    else:
        X_k_ = X_0_[..., :-1]#[:, np.newaxis]
    Y_k_ = X_0_[..., -1][:, np.newaxis]

    if maximize: Y_k_ = X_0_[..., -1][:, np.newaxis]
    else:        Y_k_ = - X_0_[..., -1][:, np.newaxis]

    # Display Initial Points
    if display:
        for i in range(Y_k_.shape[0]):
            if maximize: print('No. Init. = {} X = {} Y = {}'.format(i, X_k_[i, :], - Y_k_[i, :]))
            else:        print('No. Init. = {} X = {} Y = {}'.format(i, X_k_[i, :], Y_k_[i, :]))
    # Loop over number of itration
    for i in range(n_iterations):
        # Update Gaussian process with existing samples
        _GP.fit(X_k_, Y_k_)
        # Sample until a non nan sample is found
        while True:
            # Obtain next sampling point from the acquisition function (expected_improvement)
            x_k_1_ = __optimize(_aqf, _GP, X_k_, Y_k_, bounds_, maximize)
            # Obtain next noisy sample from the objective function
            if maximize: y_k_1_ = _f(x_k_1_, args_)
            else:        y_k_1_ = - _f(x_k_1_, args_)
            # Repeat sampling when nan is found
            if not np.isnan(y_k_1_): break
        # Display Last Sample
        if display: __diplay(i, Y_k_, x_k_1_, y_k_1_, maximize)
        # Stack sample with previous samples
        X_k_ = np.vstack((X_k_, x_k_1_))
        Y_k_ = np.vstack((Y_k_, y_k_1_))
    # Find Best Sample
    idx   = np.argmin(Y_k_)
    x_opt = X_k_[idx, :]
    # Return Optima wheter if maximazing or minimizing
    if maximize: y_opt = - Y_k_[idx]
    else:        y_opt = Y_k_[idx]
    # Dipslay the Optimal result
    if display: print('Optimal: X: {} Y: {}'.format(x_opt, y_opt))

    return x_opt, y_opt



__all__ = ['_BO_bounds', '_random_init', '_BO']
