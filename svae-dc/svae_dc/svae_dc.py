#
# Sequential Variational Autoencoder with Dynamic Compression (SVAE-DC).
#
# @contactrika
#
import logging

import numpy as np
import torch
import torch.nn as nn

import svae_dc_nets as nets


def torch_mean(lst):
    mn = np.nan if len(lst)==0 else np.array(lst).mean()
    return torch.as_tensor(mn)


class SVAE_DC(nn.Module):
    def __init__(self, pr, coder_type, latent_type, good_th, dyn_comp, device):
        super(SVAE_DC, self).__init__()
        self.latent_type = latent_type
        self.latent_seq_len = pr.K_tau
        self.y_size = pr.y_size
        self.register_buffer('xi_dim_weights', pr.xi_dim_weights)
        self.register_buffer('tau_dim_weights', torch.ones(pr.tau_size))
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('one', torch.ones(1))
        self.tau_dim_weights[-pr.y_size:] = 100.0  # y part more important
        self.tau_dim_weights /= torch.sum(self.tau_dim_weights)
        self.good_th = good_th
        self.dyn_comp = dyn_comp  # whether to use dynamic compression online
        logging.info('SVAE_DC using tau_dim_weights')
        logging.info(self.tau_dim_weights)
        self.register_buffer(
            'small_logvars_tau',
            nets.LogvarLimitScheduler.MAX_LOGVAR*torch.ones(1,pr.tau_size))
        # Generative (decoder) part of the model.
        if coder_type == 'conv':
            self.p_xi_gvn_taum = nets.ConvGauss(
                pr.tau_size-pr.y_size, pr.K_tau, pr.xi_size, pr.T_xi,
                pr.hidden_size, pr)
            self.p_y_gvn_psi = nets.LearnableGaussianDiagCondDistr(
                    pr.y_size*pr.K_tau, pr.y_size, int(pr.hidden_size/2), pr)
        else:
            print('Invalid/unimplemented encoder/decoder option', coder_type)
            assert(False)
        # Latent dynamics p(tau_k | tau{k-1}, x) and q(x|tau)
        #self.q_x_gvn_tau = nets.ConvGauss(
        #    pr.tau_size, pr.K_tau, pr.x_size, 1, pr.hidden_size*8, pr)
        if latent_type=='conv':
            self.p_tau_gvn_x = nets.ConvGauss(
                pr.x_size, 1, pr.tau_size, pr.K_tau, pr.hidden_size*4, pr,
                n_conv_layers=4)
        elif latent_type=='mlp':
            self.p_tau_gvn_x = nets.MlpGauss(
                pr.x_size, 1, pr.tau_size, pr.K_tau, pr.hidden_size*8, pr)
        else:
            print('Invalid/unimplemented latent type option', latent_type)
            assert(False)
        # Inference (encoder) part of the model.
        if coder_type == 'conv':
            self.q_tau_gvn_xi = nets.ConvGauss(
                pr.xi_size+pr.y_size, pr.T_xi,
                pr.tau_size, pr.K_tau, pr.hidden_size, pr)
        else:
            print('Invalid/unimplemented encoder/decoder option', coder_type)
            assert(False)
        self.to(device)

    def keep_only_adjustable(self):
        # Discard SVAE parts that won't be adjusted online.
        self.tau_dim_weights = None
        self.xi_dim_weights = None
        self.q_tau_gvn_xi = None
        self.p_xi_gvn_taum = None

    def encode(self, xi_1toT, bads_1toT, require_grad=False):
        assert((type(xi_1toT) == torch.Tensor) and (xi_1toT.dim() == 3))
        batch_size, _, _, = xi_1toT.size()
        # Get q(tau_{1:K} | xi_{1:T})
        inp = torch.cat([xi_1toT, bads_1toT], dim=2)
        post_q_tau_gvn_xi = self.q_tau_gvn_xi(inp)
        tau_smpl = post_q_tau_gvn_xi.sample_(require_grad).view(
            batch_size, self.latent_seq_len, -1)
        return tau_smpl, post_q_tau_gvn_xi

    def reconstruct(self, xi_1toT, bads_1toT, require_grad=False, debug=False):
        assert ((type(xi_1toT) == torch.Tensor) and (xi_1toT.dim() == 3))
        tau_smpl, post_q_tau_gvn_xi = self.encode(
            xi_1toT, bads_1toT, require_grad)
        taum_smpl, psi_smpl = self.separate_tau_psi(tau_smpl)
        recon_xi_distr = self.p_xi_gvn_taum(taum_smpl)
        recon_y_distr = self.p_y_gvn_psi(psi_smpl)
        return recon_xi_distr, recon_y_distr, tau_smpl, post_q_tau_gvn_xi

    def elbo(self, x, xi_1toT, bads_1toT, xi_std, use_laplace, gen_beta,
             debug=False):
        debug_dict = {}; debug_delta = None
        assert ((type(xi_1toT) == torch.Tensor) and (xi_1toT.dim() == 3))
        batch_size, seq_len, xi_size = xi_1toT.size()
        badness = bads_1toT.mean(dim=1).squeeze()  # mean across time

        tau_smpl, post_q_tau_gvn_xi = self.encode(
            xi_1toT, bads_1toT, require_grad=True)

        # Run reconstruction.
        taum_smpl, psi_smpl = self.separate_tau_psi(tau_smpl)
        recon_xi_distr = self.p_xi_gvn_taum(taum_smpl)
        recon_y_distr = self.p_y_gvn_psi(psi_smpl)

        # Get log p(xi_t | x_{t-1}, tau)
        # Note: we had use_laplace for our experiments. Gaussian alternative
        # has more experimental options here. Though ultimately using Laplace
        # distribution yielded faster and more reliable results for us.
        if use_laplace:
            recon_xi_distr_laplace = torch.distributions.laplace.Laplace(
                recon_xi_distr.mu, torch.exp(recon_xi_distr.logvar).sqrt())
            recon_y_distr_laplace = torch.distributions.laplace.Laplace(
                recon_y_distr.mu, torch.exp(recon_y_distr.logvar).sqrt())
            ll_recon_xi = recon_xi_distr_laplace.log_prob(
                xi_1toT.view(-1,xi_size))
            ll_recon_y = recon_y_distr_laplace.log_prob(badness.unsqueeze(1))
        else:
            # We do not standardize xi, since this would make scaling to a known
            # range problematic. Instead we use std for appropriate loss scaling
            # via passing adjust to be used when computing log_density:
            # using (x-mu)*adjust instead of (x-mu).
            adjust = torch.div(self.xi_dim_weights.view(1,-1), xi_std.view(1,-1))
            ll_recon_xi = recon_xi_distr.log_density_(
                xi_1toT.view(-1,xi_size), omit=None, adjust=adjust, debug=debug)
            ll_recon_y = recon_y_distr.log_density_(badness.unsqueeze(1))
        ll_recon_xi = ll_recon_xi.view(batch_size, seq_len, -1)
        ll_recon_xi = ll_recon_xi.sum(dim=1).sum(dim=-1)  # sum over time, xi
        ll_recon_y = seq_len*xi_size*ll_recon_y  # y as important as xi
        ll_recon = ll_recon_xi + ll_recon_y

        # Latent KL
        ptau_distr = self.p_tau_gvn_x(x)
        if use_laplace:
            ptau_distr_laplace = torch.distributions.laplace.Laplace(
                ptau_distr.mu, torch.exp(ptau_distr.logvar).sqrt())
            post_q_tau_gvn_xi_distr_laplace = torch.distributions.laplace.Laplace(
                post_q_tau_gvn_xi.mu, torch.exp(post_q_tau_gvn_xi.logvar).sqrt())
            latent_kl = torch.distributions.kl.kl_divergence(
                ptau_distr_laplace, post_q_tau_gvn_xi_distr_laplace)
        else:
            latent_kl = post_q_tau_gvn_xi.kl_to_other_distr_(ptau_distr)
        latent_kl = latent_kl.view(
            batch_size, self.latent_seq_len, -1).sum(dim=1)  # sum latent steps
        latent_kl = latent_kl.sum(dim=1)  # sum over dims, ok since diag

        # Compute generative (latent) weighting factor.
        if gen_beta is None:
            tau_size = tau_smpl.size(-1)
            gen_beta = float(seq_len*xi_size)/(self.latent_seq_len*tau_size)

        # Compute ELBO =
        # E_q[sum_{xi_t not bad} log p(xi_t | x_{t-1}, tau)] ]
        # - (E_q[ log q(tau | xi) ] + E_q[ log p(tau | x) ])
        # We multiply ELBO by -1 in the training loop to turn it into a loss.
        elbo = ll_recon - gen_beta*latent_kl

        # Make an informative debug dict.
        if debug:
            debug_dict, debug_delta = self.make_debug_dict(
                elbo, xi_1toT, badness, ll_recon_xi, ll_recon_y, recon_xi_distr,
                latent_kl, post_q_tau_gvn_xi, ptau_distr)

        # Return elbo and debug_dict
        return elbo, elbo.detach(), debug_dict, debug_delta  # done

    def make_debug_dict(self, elbo, xi_1toT, badness,
                        ll_recon_xi, ll_recon_y, recon_xi_distr,
                        latent_kl, post_q_tau_gvn_xi, ptau_distr):
        good_th = 0.1; debug_delta = None
        batch_size, seq_len, _ = xi_1toT.size()
        debug_dict = {}
        if elbo is not None: debug_dict['elbo'] = elbo.detach()
        if recon_xi_distr is not None:
            badness_np = badness.detach().cpu().numpy()
            recon_mu = recon_xi_distr.mu.view(batch_size, seq_len, -1)
            debug_delta = torch.abs(recon_mu - xi_1toT).detach().cpu().numpy()
            debug_delta = torch.from_numpy(debug_delta[badness_np<good_th])
            debug_dict['ll_recon_xi'] = ll_recon_xi.detach()
        if ll_recon_y is not None:
            debug_dict['ll_recon_y'] = ll_recon_y.detach()
        if latent_kl is not None:
            debug_dict['latent_kl'] = latent_kl.detach()
        return debug_dict, debug_delta

    def make_test_dict(self, test_x, test_xi_1toT, test_bads_1toT, test_rwd):
        test_batch_size = test_x.size(0)
        test_dict = {}
        test_badness = test_bads_1toT.mean(dim=1).squeeze()
        test_goodness = 1-test_badness
        good_th = 0.9; bad_th = 0.7
        # gen
        tau_distr = self.p_tau_gvn_x(test_x)
        _, psi_mu = self.separate_tau_psi(tau_distr.mu.view(
            test_batch_size, self.latent_seq_len, -1))
        gen_y_distr = self.p_y_gvn_psi(psi_mu)
        gen_badness = gen_y_distr.mu.squeeze()
        gen_mse = (gen_badness - test_badness).pow(2).mean()
        test_dict['test_y_gen_rmse'] = torch.sqrt(gen_mse)
        gen_goodness = 1-gen_badness
        # recon
        recon_xi_1toT_distr, recon_bads_1toT_distr, recon_tau_1toK, _ = \
            self.reconstruct(test_xi_1toT, test_bads_1toT,
                             require_grad=False, debug=False)
        recon_badness = recon_bads_1toT_distr.mu.view(
            test_batch_size, -1, 1).mean(dim=1).squeeze()  # assuming y_size==1
        recon_goodness = 1-recon_badness
        recon_mse = (recon_badness - test_badness).pow(2).mean()
        test_dict['test_y_recon_rmse'] = torch.sqrt(recon_mse)
        # num errors where predicted goodness is low, but actual is higher
        true_low_goods = test_goodness[gen_goodness<bad_th]
        test_dict['test_num_errs_gen_goodlow_true_goodhigh'] = torch.sum(
            true_low_goods>good_th).float()
        true_low_goods = test_goodness[recon_goodness<bad_th]
        test_dict['test_num_errs_recon_goodlow_true_goodhigh'] = torch.sum(
            true_low_goods>good_th).float()
        # num errors where predicted goodness is high, but actual is lower
        true_low_goods = test_goodness[gen_goodness>good_th]
        test_dict['test_num_errs_gen_goodhigh_true_goodlow'] = torch.sum(
            true_low_goods<bad_th).float()
        true_low_goods = test_goodness[recon_goodness>good_th]
        test_dict['test_num_errs_recon_goodhigh_true_goodlow'] = torch.sum(
            true_low_goods<bad_th).float()

        return test_dict

    def generate(self, x, require_grad=False, use_mean=False, debug=False):
        batch_size = x.size(0)
        # Decode xi_{1:T} sequence from the generated tau_{1:K}.
        tau_gvn_x_distr = self.p_tau_gvn_x(x)
        if use_mean:
            gen_tau_smpl = tau_gvn_x_distr.mu
        else:
            gen_tau_smpl = tau_gvn_x_distr.sample_(require_grad=require_grad)
        gen_tau_smpl = gen_tau_smpl.view(batch_size, self.latent_seq_len, -1)
        gen_taum, gen_psi = self.separate_tau_psi(gen_tau_smpl)
        xi_distr = self.p_xi_gvn_taum(gen_taum)
        y_distr = self.p_y_gvn_psi(gen_psi)
        return xi_distr, y_distr, gen_tau_smpl

    def get_latent_distr(self, x):
        return self.p_tau_gvn_x(x)

    def separate_tau_psi(self, tau):
        taum = tau[:,:,0:-self.y_size]; psi = tau[:,:,-self.y_size:]
        return taum, psi.view(-1, self.latent_seq_len*self.y_size)

    def goodness_transform(self, raw_goodness, debug=False):
        # sigmoid(x*k) to sharpen the output
        goodness = torch.sigmoid((raw_goodness-self.good_th)*25.0)
        if debug:
            logging.info('goodness')
            logging.info(raw_goodness.squeeze().detach().cpu().numpy())
            logging.info('transformed_goodness')
            logging.info(goodness.squeeze().detach().cpu().numpy())
        return goodness

    def compute_latent_state(self, x_in, debug=False):
        # Extract tau,psi from p(tau,psi|x), y from p(y|psi).
        # Set x = y*tau; grads will prop through p(tau,psi|x), p(y|psi).
        tau_distr = self.get_latent_distr(x_in)
        taum_mu, psi_mu = self.separate_tau_psi(
            tau_distr.mu.view(x_in.size(0), self.latent_seq_len, -1))
        x_tau = taum_mu.contiguous()
        if self.dyn_comp:
            y_distr = self.p_y_gvn_psi(psi_mu)
            raw_goodness = (1-y_distr.mu).unsqueeze(1)
            goodness = self.goodness_transform(raw_goodness, debug=debug)
            x_tau = goodness*x_tau
        return x_tau.view(x_in.size(0), -1), tau_distr
