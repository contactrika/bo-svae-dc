#
# NNs for Sequential Variational Autoencoder with Dynamic Compression (SVAE-DC).
#
# @contactrika
#
import logging

import numpy as np
import torch
import torch.nn as nn

from utils import prob


class LogvarFixedLimit:
    MAX_LOGVAR = 10  # STD = sqrt(exp(-10)) = 0.006

    def __init__(self, logvar_patience):
         pass

    def step(self, val):
        pass

    def logvar_limit(self):
        return LogvarFixedLimit.MAX_LOGVAR


class LogvarLimitScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    MAX_LOGVAR = 6  # STD = sqrt(exp(-6)) = 0.05

    def __init__(self, logvar_patience):
        pass
        logvar_factor = 0.9
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.logvar_optimizer = torch.optim.SGD(
            [dummy_param], lr=LogvarLimitScheduler.MAX_LOGVAR*0.99)
        super(LogvarLimitScheduler, self).__init__(
            self.logvar_optimizer, mode='max', factor=logvar_factor,
            patience=logvar_patience, cooldown=logvar_patience, min_lr=1e-5,
            threshold=1.0, threshold_mode='abs', eps=1e-08, verbose=True)

    def logvar_limit(self):
        return (LogvarLimitScheduler.MAX_LOGVAR -
                self.logvar_optimizer.param_groups[0]['lr'])


class Noop(nn.Module):
    def forward(self, x):
        return x


class NetsParams:
    def __init__(self, x_size, xi_size, T_xi, xi_dim_weights,
                 tau_size, K_tau, hidden_size, logvar_patience, debug=0):
        self.mlp_pdropout = 0.2
        self.x_size = x_size
        self.xi_size = xi_size
        self.T_xi = T_xi
        self.y_size = 1
        self.xi_dim_weights = xi_dim_weights
        self.hidden_size = hidden_size  # hidden layers for all nets
        self.tau_size = tau_size
        self.K_tau = K_tau
        # ReLU does not have hyperparameters, works with dropout and batchnorm.
        # Other options like ELU/SELU are more suitable for very deep nets
        # and have shown some promise, but no huge gains.
        # With partialVAE ReLUs will cause variance to explode on high-dim
        # inputs like pixels from image.
        # Tanh can be useful when the range needs to be restricted,
        # but saturates and trains slower.
        # ELU showed better results for high learning rates on RL experiments.
        self.nl = nn.ReLU()
        # Control latent space range.
        self.mu_nl = nn.Sigmoid()  # we need [0,1] scaled output
        self.debug = debug
        # Stabilize and speedup training by controlling minimum variance
        # adaptively (an advanced version of the idea mentioned here:
        # https://arxiv.org/abs/1805.09281).
        if logvar_patience is None:
            self.logvar_scheduler = LogvarFixedLimit(logvar_patience)
        else:
            self.logvar_scheduler = LogvarLimitScheduler(logvar_patience)


def print_debug(msg, var_dict):
    print(msg, ' ', end='')
    for key, val in var_dict.items(): print(key, val.size())


class LearnableGaussianDiagCondDistr(nn.Module):
    def __init__(self, cond_size, output_size, hidden_size, pr):
        super(LearnableGaussianDiagCondDistr, self).__init__()
        self.nn = nn.Sequential(nn.Linear(cond_size, hidden_size), pr.nl,
                                nn.Linear(hidden_size, hidden_size), pr.nl)
        # keep mu and logvar separate for easier logs (logvar -> no softplus)
        self.mu = nn.Sequential(nn.Linear(hidden_size, output_size), pr.mu_nl)
        self.logvar = nn.Sequential(nn.Linear(hidden_size, output_size))
        self.logvar_scheduler = pr.logvar_scheduler

    def forward(self, cond):
        out = self.nn(cond)
        return prob.GaussianDiagDistr(
            self.mu(out), self.logvar(out),
            self.logvar_scheduler.logvar_limit())


class LearnableGaussianDiagDistr(nn.Module):
    def __init__(self, output_size, pr):
        super(LearnableGaussianDiagDistr, self).__init__()
        # Do not use torch.augtograd.Variable to add custom params a Module:
        # https://stackoverflow.com/questions/51373919/the-purpose-of-introducing-nn-parameter-in-pytorch
        self.mu = torch.nn.Parameter(torch.zeros(1,output_size))
        self.mu_nl = pr.mu_nl
        self.logvar = torch.nn.Parameter(torch.zeros(1,output_size))
        self.logvar_scheduler = pr.logvar_scheduler

    def forward(self):
        return prob.GaussianDiagDistr(
            self.mu_nl(self.mu), self.logvar,
            self.logvar_scheduler.logvar_limit())


class ConvGauss(nn.Module):
    def __init__(self, input_size, in_seq_len, output_size, out_seq_len,
                 hidden_size, pr, n_conv_layers=3):
        assert(n_conv_layers >=3 and n_conv_layers <=4)
        super(ConvGauss, self).__init__()
        self.out_seq_len = out_seq_len
        chnls = [hidden_size, int(hidden_size/2), int(hidden_size/4)]  # deconv
        if n_conv_layers>3: chnls.append(int(hidden_size/4))
        if in_seq_len > out_seq_len: chnls.reverse()
        logging.info('Constructed ConvGauss {:d}x{:d}->{:d}x{:d} chnls:'.format(
            input_size, in_seq_len, output_size, out_seq_len))
        logging.info(chnls)
        self.nn = torch.nn.Sequential()
        l_in_out = in_seq_len
        for l in range(n_conv_layers):
            knl = 4; strd = 2
            if out_seq_len > in_seq_len:
                l_in_out, knl, strd = ConvGauss.l_out(
                    l_in_out, knl, strd, ConvGauss.l_out_deconv)
                layer = nn.ConvTranspose1d(
                    chnls[l-1] if l>0 else input_size, chnls[l], knl, strd)
            else:
                l_in_out, knl, strd = ConvGauss.l_out(
                    l_in_out, knl, strd, ConvGauss.l_out_conv)
                layer = nn.Conv1d(
                    chnls[l-1] if l>0 else input_size, chnls[l], knl, strd)
            logging.info('l_in_out {:d} knl {:d} strd {:d}'.format(
                l_in_out, knl, strd))
            self.nn.add_module('conv'+str(l), layer)
            self.nn.add_module('conv_nl'+str(l), pr.nl)
            # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            if l_in_out > 4:
                self.nn.add_module('bn'+str(l), nn.BatchNorm1d(chnls[l]))
        # Keep mu and logvar separate for easier range restrictions.
        self.mu = nn.Sequential(
            nn.Linear(l_in_out*chnls[-1], output_size*out_seq_len), pr.mu_nl)
        self.logvar = nn.Sequential(
            nn.Linear(l_in_out*chnls[-1], output_size*out_seq_len))
        self.logvar_scheduler = pr.logvar_scheduler

    @staticmethod
    def l_out_conv(l_in, knl, strd):
        dltn = 1; pad = 0
        top = l_in + 2*pad - dltn*(knl-1) - 1
        return int(top/strd + 1)

    @staticmethod
    def l_out_deconv(l_in, knl, strd):
        dltn = 1; pad = 0
        return (l_in-1)*strd - 2*pad + dltn*(knl-1) + 1

    @staticmethod
    def l_out(l_in, knl, strd, l_out_fxn):
        res = l_out_fxn(l_in, knl, strd)
        if res <= 1:
            knl = 2; strd = 1
            res = l_out_fxn(l_in, knl, strd)
        if res < 1:
            knl = 1; strd = 1
            res = l_out_fxn(l_in, knl, strd)
        return res, knl, strd

    def forward(self, in_1toN):
        assert(in_1toN.dim() in [2,3])
        if in_1toN.dim()==2: in_1toN = in_1toN.unsqueeze(1)
        batch_size, input_seq_len, data_size = in_1toN.size()
        in_1toN = in_1toN.permute(0, 2, 1)  # data_size is num channels
        out = None; idx = 0; sane_bsz = 10000  # to avoid CUDNN BatchNorm
        while idx < batch_size:    # errors (non-contiguous input wrong error)
            out_part = self.nn(in_1toN[idx:idx+sane_bsz,:,:])
            out = out_part if out is None else torch.cat([out,out_part], dim=0)
            idx += sane_bsz
        mu_out = self.mu(out.view(batch_size, -1))
        mu_out = mu_out.view(batch_size, self.out_seq_len, -1)
        logvar_out = self.logvar(out.view(batch_size, -1))
        logvar_out = logvar_out.view(batch_size, self.out_seq_len, -1)
        return prob.GaussianDiagDistr(
            mu_out.view(batch_size*self.out_seq_len, -1),
            logvar_out.view(batch_size*self.out_seq_len, -1),
            self.logvar_scheduler.logvar_limit())


class MlpGauss(nn.Module):
    def __init__(self, input_size, in_seq_len, output_size, output_seq_len,
                 hidden_size, pr):
        super(MlpGauss, self).__init__()
        logging.info('Constructed MlpGauss {:d}x{:d}->{:d}x{:d}'.format(
            input_size, in_seq_len, output_size, output_seq_len))
        self.out_seq_len = output_seq_len
        hsizes = [hidden_size]*3
        if output_seq_len < in_seq_len: hsizes.reverse()
        logging.info('MlpGauss hidden sizes'); logging.info(hsizes)
        # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.nn = nn.Sequential(
            nn.Linear(input_size*in_seq_len, hsizes[0]), pr.nl,
            nn.Dropout(p=pr.mlp_pdropout),
            nn.Linear(hsizes[1], hsizes[2]), pr.nl,
            nn.Dropout(p=pr.mlp_pdropout))
        # keep mu and logvar separate for easier logs (logvar -> no softplus)
        self.mu = nn.Sequential(
            nn.Linear(hsizes[2], output_size*output_seq_len), pr.mu_nl)
        self.logvar = nn.Sequential(
            nn.Linear(hsizes[2], output_size*output_seq_len))
        self.logvar_scheduler = pr.logvar_scheduler

    def forward(self, inp):
        batch_size = inp.size(0)
        if inp.dim()==3:  # dim1 is sequential, we will collapse it
            batch_size, input_seq_len, data_size = inp.size()
            inp = inp.view(batch_size, input_seq_len*data_size)
        out = self.nn(inp)
        mu_out = self.mu(out).view(batch_size*self.out_seq_len, -1)
        logvar_out = self.logvar(out).view(batch_size*self.out_seq_len, -1)
        res = prob.GaussianDiagDistr(
            mu_out, logvar_out, self.logvar_scheduler.logvar_limit())
        return res
