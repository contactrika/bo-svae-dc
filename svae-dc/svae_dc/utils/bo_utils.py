#
# Utils for incorporating SVAE-DC kernel into botorch.
#
# @contactrika
#
#import gc
import logging
#import os
import sys

import numpy as np
#from scipy.spatial.distance import cdist
import torch
np.set_printoptions(precision=4, threshold=sys.maxsize, suppress=True)
torch.set_printoptions(precision=4, threshold=sys.maxsize, sci_mode=False)

import gpytorch
from gpytorch.utils.transforms import inv_sigmoid
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.gen import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples

#from torch.nn.functional import softplus
#from gpytorch.utils.transforms import inv_softplus
#from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
#from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP
#from botorch import fit_gpytorch_model

from .bo_constants import GPConstants, SaneInterval
from .bo_kernels import KernelTransform, RRBFKernel, MMaternKernel, KLKernel


def printt(msg, tnsr, sci=False):
    fmt_ch = "%.6E " if sci else "%.5f "
    fmt={'float_kind':lambda x: fmt_ch % x}
    arr_str = np.array2string(tnsr.cpu().data.numpy().squeeze(), formatter=fmt)
    logging.info('{:s} {:s}'.format(msg, arr_str))


def init_and_fit_gp(train_x, train_y, y_min, y_max, device, kernel_type,
                    svae, svae_latent_dim, prev_gp=None, online_adjust=False,
                    debug=False):
    with gpytorch.settings.cg_tolerance(0.1):
        gp = GPRegressionModel(
            train_x, train_y, y_min, y_max, kernel_type, svae,
            svae_latent_dim, online_adjust, debug).to(device)
        mll = EExactMarginalLogLikelihood(gp.likelihood, gp)
        # Re-optimize hyper parameters from scratch to make sure
        # we don't get stuck in a bad local optimum for a long time.
        try:
            # Code below is the same as fit_gpytorch_model but with debug.
            # Matern kernel with default params from botorch fails, even with
            # psd_safe_cholesky() from gpytorch/utils/cholesky.py.
            mll.train()
            mll, res = fit_gpytorch_scipy(
                mll, method='L-BFGS-B', track_iterations=True,
                options={'maxiter': GPConstants.MAX_MLL_FIT_ITER})
            lls = np.array([-info.fun for info in res]) # info.fun has loss=-ll
            logging.info('fit_gpytorch_scipy lls'); logging.info(lls)
            mll.eval()
        except RuntimeError as e:
            logging.info('WARNING: fit_gpytorch_scipy threw RuntimeError:')
            logging.info(e)
            logging.info('Try to load hypers from previous trial instead')
            if prev_gp is not None: gp.load_state_dict(prev_gp.state_dict())
        except ValueError as e:
            # gpytorch/functions/_inv_quad_log_det.py sometimes throws:
            # preconditioner, precond_lt, logdet_correction = lazy_tsr.detach()._preconditioner()
            # ValueError: not enough values to unpack (expected 3, got 2)
            # Likely because of gpytorch/lazy/added_diag_lazy_tensor.py:62:
            # UserWarning: NaNs encountered in preconditioner computation.
            # Attempting to continue without preconditioning.
            # "NaNs encountered in preconditioner computation.
            logging.info('WARNING: fit_gpytorch_scipy threw ValueError:')
            logging.info(e)
            logging.info('Try to load hypers from previous trial instead')
            if prev_gp is not None: gp.load_state_dict(prev_gp.state_dict())
    return mll, gp


def make_kernel(kernel_type, ard_dims, svae=None):
    kernel_transform = None
    if svae is not None:
        assert(kernel_type in ['KL', 'uKL', 'LKL'] or
               kernel_type.startswith('SVAE'))
        latent = False if kernel_type=='KL' else True
        kernel_transform = KernelTransform(svae, latent=latent)
    min_outsc = GPConstants.MIN_OUTPUTSCALE
    max_outsc = GPConstants.MAX_OUTPUTSCALE
    hypers_unrestricted = (kernel_type=='uSE' or kernel_type=='uMatern' or
                           kernel_type=='uKL')
    if hypers_unrestricted:
        min_outsc = GPConstants.MIN_UNRESTR_OUTPUTSCALE
        max_outsc = GPConstants.MAX_UNRESTR_OUTPUTSCALE
    outsc_constr = SaneInterval(
        lower_bound=torch.as_tensor(min_outsc),
        upper_bound=torch.as_tensor(max_outsc),
        transform=torch.sigmoid,
        initial_value=inv_sigmoid(torch.as_tensor(0.9999/max_outsc)))  # ~1.0
    if kernel_type.endswith('SE'):
        covar_module = gpytorch.kernels.ScaleKernel(
            RRBFKernel(ard_dims, kernel_transform, hypers_unrestricted),
            outputscale_constraint=outsc_constr)
    elif kernel_type.endswith('Matern'):
        covar_module = gpytorch.kernels.ScaleKernel(
            MMaternKernel(ard_dims, kernel_transform),
            outputscale_prior=GammaPrior(2.0, 0.15),
            outputscale_constraint=outsc_constr)
    elif kernel_type.endswith('KL'):  # symmetrized KL (in latent space for LKL)
        assert(kernel_type=='KL' or kernel_type=='uKL' or kernel_type=='LKL')
        covar_module = gpytorch.kernels.ScaleKernel(
            KLKernel(kernel_transform), outputscale_constraint=outsc_constr)
    else:
        logging.error('Unsupported kernel type', kernel_type)
        assert(False)
    return covar_module, kernel_transform


class GPRegressionModel(SingleTaskGP):
    def __init__(self, train_x, train_y_raw, y_min, y_max, kernel_type,
                 svae=None, svae_latent_dim=None, online_adjust=False,
                 debug=False):
        self.debug = debug
        # SingleTaskGP is supposed to work best with standardized y
        # (per botorch docs), but in fact this gives poor results for
        # non-trivial costs. So we scale costs to be in [0,1] instead.
        self.y_min = y_min; self.y_max = y_max
        self.scale_y_fxn = lambda y: (y-self.y_min)/(self.y_max-self.y_min)
        self.unscale_y_fxn = lambda y: y*(self.y_max-self.y_min)+self.y_min
        train_y = self.scale_y_fxn(train_y_raw)
        #
        # Init Likelihood
        #
        # Could initialize custom likelihood hyperparams, as in example below:
        #noise_prior = GammaPrior(1.1, 0.05)
        #noise_prior_mode = (noise_prior.concentration-1.0)/noise_prior.rate
        #noise_contraint = gpytorch.constraints.constraints.GreaterThan(
        #    GPConstants.MIN_INFERRED_NOISE_VAR,
        #    transform=None, initial_value=noise_prior_mode)
        #lik = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(
        #    noise_prior=noise_prior, noise_constraint=noise_contraint)
        noise_contraint = SaneInterval(
            lower_bound=torch.as_tensor(GPConstants.MIN_INFERRED_NOISE_VAR),
            upper_bound=torch.as_tensor(GPConstants.MAX_INFERRED_NOISE_VAR),
            transform=torch.sigmoid)
        lik = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(
            noise_constraint=noise_contraint)
        super(GPRegressionModel, self).__init__(train_x, train_y, lik)
        #super(GPRegressionModel, self).__init__(train_x, train_y)
        #train_y_var = torch.ones_like(train_y)*GPConstants.FIXED_NOISE_VAR
        #super(GPRegressionModel, self).__init__(train_x, train_y, train_y_var)
        #
        # Init Mean
        #
        #self.mean_module = gpytorch.means.ConstantMean()
        # Reasonable prior on the mean would indicate what we know about the
        # range of objective values. This *should* be set with meaningful values
        # if we have a very small budget of initial (random) and overall trials.
        # From gpytorch/test/examples/test_kissgp_variational_regression.py
        self.mean_module = gpytorch.means.ConstantMean(
            prior=gpytorch.priors.SmoothedBoxPrior(
                GPConstants.OBJECTIVE_LOW, GPConstants.OBJECTIVE_HIGH))
        #
        # Init Kernel
        #
        self.kernel_type = kernel_type
        kernel_x_size = train_x.size(1)
        if svae_latent_dim is not None:
            kernel_x_size = svae_latent_dim
        self.covar_module, self.kernel_transform = make_kernel(
            kernel_type, kernel_x_size, svae)
        # Make sure gradients are propagated through latent NNs
        # This will happen automatically if we add the NNs to the GP model here.
        # For settings with a larger number of trials can have self.svae = svae.
        # We consider only 10-20 trials, so only adjust mu of psi->y NN online.
        if svae is not None:
            if online_adjust:
                logging.info('SVAE psi->y params will be adjusted online')
                self.svae_p_y_gvn_psi_mu = svae.p_y_gvn_psi.mu
            else:
                logging.info('SVAE params kept fixed')

    def forward(self, x):
        #if ((self.kernel_transform is not None) and
        #    self.kernel_type.endswith(('SE', 'Matern'))):
        #    #x, _ = self.kernel_transform.apply(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        output = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return output

    def print_kernel_transform_debug(self, x):
        if self.covar_module.base_kernel.kernel_transform is not None:
            self.covar_module.base_kernel.kernel_transform.apply(x, debug=True)
        #if hasattr('self', 'svae'):
            #print('print_kernel_transform: svae.p_tau_gvn_x.nn.conv0.weight',
            #      self.svae.p_tau_gvn_x.nn.conv0.weight[0:5,0:2,0:2])
            #print('svae.p_y_gvn_psi ',
            #      self.svae.p_y_gvn_psi.nn[0].weight[0:5,0:5])
        if hasattr(self, 'svae_p_y_gvn_psi_mu'):
            for idx, mdl in enumerate(self.svae_p_y_gvn_psi_mu.modules()):
                if idx>0: break
                print('svae_p_y_gvn_psi_mu', idx, '->', mdl)
                for seq_idx, seq_mdl in enumerate(mdl):
                    print('seq_mdl', seq_idx, '->', seq_mdl)
                    if hasattr(seq_mdl, 'bias'):
                        print('bias[0:5]', seq_mdl.bias[0:5])
                    if hasattr(seq_mdl, 'weight'):
                        print('weight[0:5,0:5]', seq_mdl.weight[0:5,0:5])

    def print_predict(self, x, y):
        if self.debug: self.print_kernel_transform_debug(x)
        if y.size(0) > 50:
            perm = torch.randperm(y.size(0)); perm = perm[0:5]
            x = x[perm]; y = y[perm]  # take a random subset of the given data
        scaled_y = self.scale_y_fxn(y)
        check_x = x+0.01  # points near to those appearing in training data
        check_x = check_x.clamp(0,1)
        try:
            post = self.posterior(check_x)
        except RuntimeError as e:
            logging.info('GP posterior threw RuntimeError')
            logging.info(e)
            logging.info('Skip printing for now')
            return
        post_y = self.unscale_y_fxn(post.mean)
        printt('print_predict(): actual y', y)
        printt('predicted y (for nearby x)', post_y)
        printt('scaled actual y', scaled_y)
        printt('scaled predicted y', post.mean)
        printt('scaled predicted std', post.variance.sqrt())
        if hasattr(self.covar_module, 'outputscale'):
            printt('outputscale', self.covar_module.outputscale)
        if (hasattr(self.covar_module, 'base_kernel') and
            hasattr(self.covar_module.base_kernel, 'lengthscale') and
            self.covar_module.base_kernel.lengthscale is not None):
            printt('lengthscale', self.covar_module.base_kernel.lengthscale)
            #printt('raw_lengthscale',
            #       self.covar_module.base_kernel.raw_lengthscale)


class EExactMarginalLogLikelihood(ExactMarginalLogLikelihood):
    # Disable checking for our NN-based parts of the model.
    def named_parameters_and_constraints(self):
        for name, param in self.named_parameters():
            yield name, param, self.constraint_for_parameter_name(name)

    def constraint_for_parameter_name(self, param_name):
        if 'p_tau_gvn_x' in param_name or 'p_y_gvn_psi': return None
        return super(GPRegressionModel).constraint_for_parameter_name(
            param_name)


def optimize_UCB(acq_function, x_bounds, unscale_y_fxn, seed):
    """Optimize UCB using random restarts"""
    # The alternative would be to call botorch.optim.joint_optimize().
    # However, that function is rather specialized for EI.
    # Hence a more custom option for ensuring good performance for UCB.
    x_init_batch_all = None; y_init_batch_all = None
    for rnd in range(GPConstants.N_RESTART_ROUNDS):
        # TODO: init/pass seed to draw_sobol_samples() for reproducibility.
        x_rnd = draw_sobol_samples(
            bounds=x_bounds, seed=seed,
            n=GPConstants.N_RESTART_CANDIDATES, q=1)
        # The code below is like initialize_q_batch(), but with stability checks.
        # botorch.optim.initialize_q_batch() also does a few more hacks like:
        # max_val, max_idx = torch.max(y_rnd, dim=0)
        # if max_idx not in idcs: idcs[-1] = max_idx # make sure we get the maximum
        # These hacks don't seem to help the worst cases, so we don't include them.
        x_init_batch = None
        y_init_batch = None
        try:
            with torch.no_grad():
                y_rnd = acq_function(x_rnd)
            finite_ids = torch.isfinite(y_rnd)
            x_rnd_ok = x_rnd[finite_ids]
            y_rnd_ok = y_rnd[finite_ids]
            y_rnd_std = y_rnd.std()
            if torch.isfinite(y_rnd_std) and y_rnd_std > GPConstants.MIN_STD:
                z = y_rnd - y_rnd.mean() / y_rnd_std
                weights = torch.exp(1.0*z)
                bad_weights = (torch.isnan(weights).any() or
                               torch.isinf(weights).any() or
                               (weights<0).any() or weights.sum() <= 0)
                if not bad_weights:
                    idcs = torch.multinomial(weights,
                                             GPConstants.N_RESTARTS_PER_ROUND)
                    x_init_batch = x_rnd_ok[idcs]
                    y_init_batch = y_rnd_ok[idcs]
            if x_init_batch is None and x_rnd_ok.size(0)>0:
                idcs = torch.randperm(n=x_rnd_ok.size(0))
                x_init_batch = x_rnd_ok[idcs][:GPConstants.N_RESTARTS_PER_ROUND]
                y_init_batch = y_rnd_ok[idcs][:GPConstants.N_RESTARTS_PER_ROUND]
        except RuntimeError as e:
            logging.info('WARNING: acq_function threw RuntimeError:')
            logging.info(e)
        if x_init_batch is None: continue  # GP-based queries failed
        if x_init_batch_all is None:
            x_init_batch_all = x_init_batch; y_init_batch_all = y_init_batch
        else:
            x_init_batch_all = torch.cat([x_init_batch_all, x_init_batch], dim=0)
            y_init_batch_all = torch.cat([y_init_batch_all, y_init_batch], dim=0)
    # If GP-based queries failed (e.g. Matern Cholesky failed) use random pts.
    if x_init_batch_all is None:
        logging.info('WARNING: all acq_function tries failed; sample randomly')
        nrnd = GPConstants.N_RESTARTS_PER_ROUND*GPConstants.N_RESTART_ROUNDS
        x_init_batch_all = draw_sobol_samples(bounds=x_bounds, n=nrnd, q=1)
    else:
        # Print logs about predicted y of the points.
        y_init_batch_all_sorted, _ = y_init_batch_all.sort(descending=True)
        logging.info('optimize_UCB y_init_batch_all scaled')
        logging.info(y_init_batch_all.size())
        logging.info(y_init_batch_all_sorted)
        y_init_batch_all_unscaled, _ = unscale_y_fxn(
            y_init_batch_all).sort(descending=True)
        logging.info('optimize_UCB y_init_batch_all unscaled')
        logging.info(y_init_batch_all_sorted.size())
        logging.info(y_init_batch_all_unscaled)
    # ATTENTION: gen_candidates_scipy does not clean up GPU tensors, memory
    # usage sometimes grows, and gc.collect() does not help.
    # TODO: Need to file a botorch bug.
    try:
        batch_candidates, batch_acq_values = gen_candidates_scipy(
            initial_conditions=x_init_batch_all,
            acquisition_function=acq_function,
            lower_bounds=x_bounds[0], upper_bounds=x_bounds[1],
            options={'maxiter': GPConstants.MAX_ACQ_ITER,
                     'method': 'L-BFGS-B'})  # select L-BFGS-B for reasonable speed
        assert(torch.isfinite(batch_candidates).all())
        #remove_too_close(gp_model, batch_candidates, batch_acq_values)
        # same code as in botorch.gen.get_best_candidates()
        best_acq_y, best_id = torch.max(batch_acq_values.view(-1), dim=0)
        next_x = batch_candidates[best_id].squeeze(0).detach()
    except RuntimeError as e:
        logging.info('WARNING: gen_candidates_scipy threw RuntimeError:')
        logging.info(e)
        next_x = x_init_batch_all[0].squeeze()
        best_acq_y = torch.tensor(-1000000.0).to(device=next_x.device)
    # Check x is within the [0,1] boundaries.
    if (next_x<0).any() or (next_x>1).all() or (next_x!=next_x).any():
        print('WARNING: GP optimization returned next_x', next_x)
        next_x = torch.zeros_like(next_x)
    return next_x, best_acq_y


class UpperConfidenceBound(AnalyticAcquisitionFunction):
    # Code from botorch.acquisition; added sanity checks and debug prints.
    def __init__(self, model, beta, maximize=True) -> None:
        super().__init__(model=model)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    def forward(self, X):
        self.beta = self.beta.to(X)
        batch_shape = X.shape[:-2]
        posterior = self.model.posterior(X)
        self._validate_single_output_posterior(posterior)
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        # TODO: internal GP code can return <0 variances. Need to file a bug.
        if (variance<=0).any():
            msg = 'ERROR: invalid variance {:0.4f} in GP posterior'
            logging.error(msg.format(variance.min()))
            # assert(False)
        # if not torch.isfinite(delta).all(): assert(False)
        return mean+delta if self.maximize else mean-delta


def remove_too_close(gp_model, batch_candidates, batch_acq_values):
    # Remove candidates that are too close to points already evaluated (X)
    # (replace them by completely random points).
    # This could prevent some Cholesky decomposition failures.
    X = gp_model.train_inputs[0]
    dists = torch.cdist(batch_candidates.squeeze(), X)
    assert(torch.isfinite(dists).all())
    min_dists = dists.min(dim=1)[0]
    min_dists = torch.clamp(min_dists - GPConstants.MIN_DIST_TO_EVALUAED_X,
                            0, dists.max())
    ok_ids = torch.nonzero(min_dists)
    if (batch_candidates.size(0) > ok_ids.size(0)):
        logging.info('optimize_UCB removed {:d} candidates close to X'.format(
            batch_candidates.size(0)-ok_ids.size(0)))
        return batch_candidates[ok_ids], batch_acq_values[ok_ids]
    else:
        return batch_candidates, batch_acq_values
