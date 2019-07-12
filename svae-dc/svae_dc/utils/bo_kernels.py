#
# Custom GP kernels for Bayesian Optimization.
#
# @contactrika
#
import logging
import sys
import time

import torch
import gpytorch

from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.utils.transforms import inv_sigmoid

from .prob import GaussianDiagDistr
from .bo_constants import GPConstants, SaneInterval


def assert_isclose(tnsr1, tnsr2, eps=1e-6):
    ok = torch.all(torch.lt(torch.abs(torch.add(tnsr1, -tnsr2)), eps))
    if not ok:
        torch.set_printoptions(precision=12, threshold=sys.maxsize,
                               sci_mode=False)
        print('assert_isclose(): tnsr1\n', tnsr1)
        print('vs tnsr2\n', tnsr2)
        print('diff', torch.abs(torch.add(tnsr1, -tnsr2)))
        assert(False)


class KernelTransform:
    def __init__(self, tau_fxn, latent=True):
        self.tau_fxn = tau_fxn
        self.latent = latent

    def apply(self, x, debug=False):
        # x: [n x d] or [b x n x d]
        orig_batch_size = x.size(0); x_dim = x.size(-1)
        x_in = x.view(-1, x_dim)
        if self.latent:
            out_x, out_distr = self.tau_fxn.compute_latent_state(x_in, debug)
        else:
            # In theory, we could samples from SVAE to get an estimate of the
            # variance. But in practice 1) VI is prone to under-estimating
            # variance, so we won't get precise estimates even with larger
            # number of samples and 2) doing sampling a large number of times
            # for each point in the kernel query becomes prohibitively
            # expensive. So here we take the mean and a reasonable fixed
            # variance, so that we can give this approach a chance. The approach
            # of taking KL in the original space is overall unlikely to work
            # for long trajectories, but need to compare against it anyway.
            #out_distr, _, _ = self.tau_fxn.generate(
            #        x_in, require_grad=True, use_mean=True, debug=debug)
            #out_distr = GaussianDiagDistr(
            #    out_distr.mu, out_distr.logvar, logvar_limit=2)
            #out_x = out_distr.mu
            out_distr, _, _ = self.tau_fxn.generate(
                    x_in, require_grad=True, use_mean=False, debug=debug)
            x_smpl = out_distr.sample_(require_grad=True)
            m = torch.zeros_like(x_smpl)
            S = torch.zeros_like(x_smpl)
            n = 0
            nsamples = 4
            for k in range(nsamples):
                prev_mean = m
                n = n + 1
                if k>0:
                    out_distr, _, _ = self.tau_fxn.generate(
                        x_in, require_grad=False, debug=debug)
                    x_smpl = out_distr.sample_(require_grad=False)
                m = m + (x_smpl-m)/n
                S += (x_smpl-m)*(x_smpl-prev_mean)
            out_x_mean = m
            out_x_std = torch.sqrt(S/(n-1))
            out_x_logvar = torch.log(torch.pow(out_x_std,2))
            logvar_limit = 4  # since VI is prone to under-estimating variance
            out_distr = GaussianDiagDistr(
                out_x_mean, out_x_logvar, logvar_limit)
            out_x = out_x_mean
        if x.dim()==3:
            other_size = x.size(1)
            out_x = out_x.view(orig_batch_size, other_size, -1)
        else:
            out_x = out_x.view(orig_batch_size, -1)
        return out_x, out_distr


class KLKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel_transform):
        super(KLKernel, self).__init__()
        self.kernel_transform = kernel_transform

    @staticmethod
    def postprocess_dists(dists):
        # Make the kernel PSD (as in BBK papers; 2 doesn't matter since will
        # wrap ScaleKernel around this one to optimize the output scale).
        return dists.div_(-2).exp_()

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if diag and torch.equal(x1, x2):
            return self.postprocess_dists(torch.zeros_like(x1))  # KL to self=0
        # gpytorch has strange conventions on how x1,x2 are given in batch mode:
        # if x1!=x2, then x1 is [batch x 1 x d], x2 is [batch x n_train+1 x d],
        # where 'batch' is the number of points we want to query (e.g.
        # 'acq_batch' during acquisition function optimization).
        # To avoid redundant NN forward calls we need inputs of the form:
        # x1=[acq_batch x d], x2=[n_train x d]. So we parse inputs in this way.
        assert(last_dim_is_batch==False)
        batch_mode = False
        if x1.dim()==3:
            batch_mode = True
            x1, x2 = self.get_core(x1, x2)  # make x1, x2 [n x d], [m x d]
        out = self.compute_KL_dists(x1, x2)
        if batch_mode:
            # Add a column to out with zeros indicating KL of queries with self.
            # This is what gpytorch expects when it sends batch mode queries.
            self_kl = torch.zeros(out.size(0), 1, device=out.device)
            out = torch.cat([out, self_kl], dim=1)
            out = out.unsqueeze(1)
        res = self.postprocess_dists(out)
        return res

    def get_core(self, x1, x2):
        assert(x1.size(1)==1)
        assert(x2.dim()==3)
        # check that the last entry of x2's middle dimension indeed
        # simply repeats the points that are already given by x1
        assert_isclose(x1[:,0,:], x2[:,-1,:])
        # get info out of x1 and x2 and transform these to [n x d], [m x d]
        x1 = x1.squeeze()
        ntrain = x2.size(1)-1
        x2 = x2[0,0:ntrain,:]
        return x1, x2

    def compute_KL_dists(self, x1, x2):
        assert(x1.dim()==2)
        assert(x2.dim()==2)
        # After the above parsing we will be working in non-batch mode:
        # x1 is [n x d], x2 is [m x d]
        # Get NN output distributions for train and query points.
        _, tau1_distrs = self.kernel_transform.apply(x1)
        _, tau2_distrs = self.kernel_transform.apply(x2)
        tau1_mus = tau1_distrs.mu.view(x1.size(0), -1)
        tau1_logvars = tau1_distrs.logvar.view(x1.size(0), -1)
        tau2_mus = tau2_distrs.mu.view(x2.size(0), -1)
        tau2_logvars = tau2_distrs.logvar.view(x2.size(0), -1)
        out = torch.zeros(x1.size(0), x2.size(0)).to(device=x1.device)
        # Ok to have an explicit loop here, since ntrain will always be small
        # for use cases we are interested in (this is the number of BO trials
        # on the real system, which are assumed to be expensive to run/eval).
        for tr in range(x2.size(0)):
            tau2_mu = tau2_mus[tr].unsqueeze(0)
            tau2_logvar = tau2_logvars[tr].unsqueeze(0)
            kl_val = GaussianDiagDistr.symmetrized_kl_between(
                tau1_mus, tau1_logvars, tau2_mu, tau2_logvar, debug=False)
            #if out.size(0)<20:
            #    print('-------------------- KL kernel --------------------')
            #    print('kl_val', kl_val)
            #    print('x1[0]', x1[0])
            #    print('x2[0]', x2[0])
            #    print('tau1_mus[0,0:5]', tau1_mus[0,0:5])
            #    print('tau2_mu[0,0:5]', tau2_mu[0,0:5])
            #    print('tau1_logvars[0,0:5]', tau1_logvars[0,0:5])
            #    print('tau2_logvar[0,0:5]', tau2_logvar[0,0:5])
            assert(kl_val.size(0)==out.size(0))
            assert(kl_val.size(1)==1)
            out[:,tr] = kl_val[:,0]
        return out

    # A slow but straightforward implementation for comparison to make sure
    # that the more efficient implementations run as expected.
    def dbgforward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert(last_dim_is_batch==False)
        if diag and torch.equal(x1, x2):
            return self.postprocess_dists(torch.zeros_like(x1))  # KL to self=0
        was_batch_mode = True
        if x1.dim()==2 or x2.dim()==2:
            was_batch_mode = False
            if x1.dim()==2: x1 = x1.unsqueeze(0)
            if x2.dim()==2: x2 = x2.unsqueeze(0)
        bsz, n, d = x1.size(); m = x2.size(1)
        out = torch.zeros(bsz, n, m).to(device=x1.device)
        for bid in range(bsz):
            for nid in range(n):
                for mid in range(m):
                    _, tau1_dist = self.kernel_transform.apply(
                        x1[bid, nid, :].unsqueeze(0))
                    _, tau2_dist = self.kernel_transform.apply(
                        x2[bid, mid, :].unsqueeze(0))
                    kl_val = GaussianDiagDistr.symmetrized_kl_between(
                        tau1_dist.mu.view(1, -1), tau1_dist.logvar.view(1, -1),
                        tau2_dist.mu.view(1, -1), tau2_dist.logvar.view(1, -1))
                    out[bid, nid, mid] = kl_val
        if not was_batch_mode: out = out.squeeze(0)
        return self.postprocess_dists(out)

    def tstforward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        strt = time.time()
        res = self.fforward(x1, x2, diag, last_dim_is_batch, **params)
        print('LKLKernel.fforward took', time.time()-strt)
        strt = time.time()
        dbgres = self.dbgforward(x1, x2, diag, last_dim_is_batch, **params)
        print('LKLKernel.dbgforward took', time.time()-strt)
        assert_isclose(res, dbgres, eps=1e-4)
        return res


class RRBFKernel(gpytorch.kernels.RBFKernel):
    def __init__(self, ard_dims, kernel_transform, hypers_unrestricted=False):
        #l_constr = gpytorch.constraints.Positive(
        #    initial_value=inv_softplus(torch.as_tensor(1.0)))
        min_lsc = GPConstants.MIN_LENGTHSCALE
        max_lsc = GPConstants.MAX_LENGTHSCALE
        if hypers_unrestricted:
            min_lsc = GPConstants.MIN_UNRESTR_LENGTHSCALE
            max_lsc = GPConstants.MAX_UNRESTR_LENGTHSCALE
        l_constr = SaneInterval(
            lower_bound=torch.as_tensor(min_lsc),
            upper_bound=torch.as_tensor(max_lsc),
            transform=torch.sigmoid,
            initial_value=inv_sigmoid(torch.as_tensor(0.9999/max_lsc)))  # ~1.0
        #l_prior = gpytorch.priors.SmoothedBoxPrior(
        #    GPConstants.LENGTHSCALE_LOW,
        #    GPConstants.LENGTHSCALE_HIGH,
        #    sigma=GPConstants.LENGTHSCALE_PRIOR_SIGMA)
        super(RRBFKernel, self).__init__(
            ard_num_dims=ard_dims, lengthscale_constraint=l_constr)
        self.kernel_transform = kernel_transform

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert(last_dim_is_batch==False)
        if self.kernel_transform is not None:
            x1, _ = self.kernel_transform.apply(x1)
            x2, _ = self.kernel_transform.apply(x2)
        return super(RRBFKernel, self).forward(x1, x2, diag, **params)


class MMaternKernel(gpytorch.kernels.MaternKernel):
    def __init__(self, ard_dims, kernel_transform):
        # Default values from botorch/models/gp_regression.py
        # GammaPrior(3.0, 6.0)) means alpha=3, beta=6
        # nu=2.5 means Matern 5/2 kernel (1/2, 3/2, 5/2)
        # E[X~Gamma(alpha,beta)] = alpha/beta
        # Var[X~Gamma(alpha,beta)] = alpha/(beta^2)
        core_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=ard_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0), param_transform=None)
        super(MMaternKernel, self).__init__()
        self.kernel_transform = kernel_transform

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert(last_dim_is_batch==False)
        if self.kernel_transform is not None:
            x1, _ = self.kernel_transform.apply(x1)
            x2, _ = self.kernel_transform.apply(x2)
        return super(MMaternKernel, self).forward(x1, x2, diag, **params)
