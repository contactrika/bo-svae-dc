#
# Useful constants for GPs and Bayesian Optimization.
#

import gpytorch


class GPConstants(object):
    # For hardware runs we did (for Daisy were running on a laptop):
    # N_RESTART_CANDIDATES=2K, N_RESTARTS_PER_ROUND=5, N_RESTART_ROUNDS=5
    #
    # If you are running into cuda memory issues: this is a known problem with
    # BoTorch/cuda memory interplay, but will be fixed soon by the botorch team.
    # Meanwhile you can try using smaller:
    # N_RESTART_CANDIDATES, N_RESTARTS_PER_ROUND, N_RESTART_ROUNDS
    MAX_MLL_FIT_ITER = 100  # max iterations to fit marginal LL (hyp opt) (50)
    MAX_ACQ_ITER = 10  # max iterations to optimize acquisition function (100)
    N_RESTART_CANDIDATES = 1000 #2000  # num items from which to select restarts (2K)
    N_RESTARTS_PER_ROUND = 5     # number of restarts for acquisition optimiz:
    N_RESTART_ROUNDS = 40 #20        # N_RESTARTS_PER_ROUND*N_RESTARTS_ROUNDS
    MIN_STD = 1e-6  # minimum standard deviation for standardization
    # Meaningful choices for initialization of hyperparameter priors.
    # We are standardizing Y, so prior mainly over [0,1].
    OBJECTIVE_LOW = -0.0
    OBJECTIVE_HIGH = 1.0
    MIN_UNRESTR_OUTPUTSCALE = 0.0001
    MAX_UNRESTR_OUTPUTSCALE = 100.0
    MIN_OUTPUTSCALE = 0.1  # 0.0001  # <0.001 can make mll overfit
    MAX_OUTPUTSCALE = 1.0  # 100.0
    MIN_UNRESTR_LENGTHSCALE = 0.0001
    MAX_UNRESTR_LENGTHSCALE = 100.0
    MIN_LENGTHSCALE = 0.01 # 0.0001
    MAX_LENGTHSCALE = 10.0 # 100.0
    MIN_INFERRED_NOISE_VAR = 1e-6  # 1-e6 in botorch/models/gp_regression.py
    MAX_INFERRED_NOISE_VAR = 0.1
    #MIN_DIST_TO_EVALUAED_X = 0.0001  # trying to avoid Cholesky decomp failures


class SaneInterval(gpytorch.constraints.Interval):
    # gpytorch.constraints.Interval.transform() does something strange:
    # transformed_tensor = (_transform(tensor) * upper_bound) + lower_bound
    # instead of multiplying by range=upped_bound-lower_bound
    def transform(self, tensor):
        if not self.enforced: return tensor
        range = self.upper_bound - self.lower_bound
        transformed_tensor = (self._transform(tensor)*range) + self.lower_bound
        return transformed_tensor


def reward_lims(env_name):
    if env_name.startswith('Daisy'): return [-200.0, 200.0]
    if env_name.startswith(('Yumi', 'Franka')): return [-2.0, 2.0]
    if env_name.startswith('Sawyer'): return [0.0, 1.0]
