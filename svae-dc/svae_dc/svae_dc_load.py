#
# Utility for loading trained SVAE-DC model.
#
# @contactrika
#
import logging
import os

import torch

from svae_dc_nets import NetsParams
from svae_dc import SVAE_DC


def svae_dc_load(env, policy, svae_dc_checkpt, override_good_th, env_name,
                 device, dyn_comp=True, load_only_adjustable=True):
    checkpt_file = os.path.expanduser(svae_dc_checkpt)
    assert(os.path.isfile(checkpt_file))
    logging.info("=> loading checkpoint '{}'".format(checkpt_file))
    chkpt = torch.load(checkpt_file, map_location=device)
    svae_dc_args = chkpt['all_args'][0]
    logging.info('svae_dc_args'); logging.info(svae_dc_args)
    x_size = policy.get_params().shape[0]
    xi_size = env.observation_space.shape[0]
    if 'Daisy' in env_name:
        xi_dim_weights = torch.from_numpy(
            env.dim_weights).float().to(device)
        logging.info('Using xi_dim_weights'); logging.info(xi_dim_weights)
    else:
        xi_dim_weights = torch.ones(xi_size, device=device)
    T_xi = svae_dc_args.max_episode_steps+1
    K_tau = int(svae_dc_args.svae_dc_K_tau_scale*T_xi)
    svae_dc_params = NetsParams(
        x_size, xi_size, T_xi, xi_dim_weights,
        svae_dc_args.svae_dc_tau_size, K_tau,
        svae_dc_args.svae_dc_hidden_size,
        svae_dc_args.svae_dc_logvar_patience, debug=False)
    logging.info('SVAE_DC params:'); logging.info(svae_dc_params.__dict__)
    good_th = svae_dc_args.svae_dc_good_th
    if override_good_th is not None: good_th = override_good_th
    svae_dc = SVAE_DC(pr=svae_dc_params,
                      coder_type=svae_dc_args.svae_dc_coder_type,
                      latent_type=svae_dc_args.svae_dc_latent_type,
                      good_th=good_th, dyn_comp=dyn_comp, device=device)
    svae_dc.load_state_dict(chkpt['svae_dc_state_dict'])
    if load_only_adjustable: svae_dc.keep_only_adjustable()
    svae_dc.to(device)
    svae_dc.eval()  # set internal pytorch flags
    logging.info('Loaded with dyn_comp {:d} load_only_adjustable {:d}'.format(
        dyn_comp, load_only_adjustable))
    kernel_transform = svae_dc
    kernel_transform_dim = svae_dc_params.K_tau*(
            svae_dc_params.tau_size-svae_dc_params.y_size)
    return kernel_transform, kernel_transform_dim
