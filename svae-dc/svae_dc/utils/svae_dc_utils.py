#
# Utils for Sequential VAE with Dynamic Compression (SVAE-DC).
#
# @contactrika
#
from glob import glob
import logging
import os

import numpy as np
import torch
from torch.utils import data
torch.set_printoptions(precision=4, linewidth=150, threshold=None, sci_mode=False)

import gym

from .policy_classes import StructuredPolicy


class SVAEDataset(data.Dataset):

    def __init__(self, env_episodes_file):
        files = glob(os.path.expanduser(env_episodes_file))
        assert(len(files)>0)
        for fl in files: assert(os.path.exists(fl))
        ids = torch.randperm(len(files))
        x_lst = []; xi_lst = []; bads_lst = []; rwd_lst = []
        num_pts_loaded = 0
        for idx in ids:
            x, xi, bads, rwd = SVAEDataset.load_file(files[idx])
            x_lst.append(x); xi_lst.append(xi)
            bads_lst.append(bads); rwd_lst.append(rwd)
            num_pts_loaded += x.size(0)
            print('loaded', num_pts_loaded, 'so far')
            if num_pts_loaded >= 50000: break  # 50K pts on each train pass
        self.x = torch.cat(x_lst, dim=0)
        self.xi_1toT = torch.cat(xi_lst, dim=0)
        self.bads_1toT = torch.cat(bads_lst, dim=0)
        self.rwd = torch.cat(rwd_lst, dim=0)
        self.xi_std = SVAEDataset.compute_xi_std(self.xi_1toT)

    def __getitem__(self, idx):
        return self.x[idx], self.xi_1toT[idx], \
               self.bads_1toT[idx], self.rwd[idx]

    def __len__(self):
        return self.x.size(0)

    @staticmethod
    def load_file(env_episodes_file):
        logging.info('loading {:s}'.format(env_episodes_file))
        episodes_data = np.load(env_episodes_file)
        x = torch.from_numpy(episodes_data['x_buf']).float()
        xi_1toT = torch.from_numpy(episodes_data['xi_1toT_buf']).float()
        bads_1toT = torch.from_numpy(episodes_data['bads_1toT_buf']).float()
        tot, seq_len, _ = xi_1toT.size()
        if 'rwd_buf' in episodes_data.keys():
            rwd = torch.from_numpy(episodes_data['rwd_buf']).float()
        else:
            rwd = torch.zeros(tot,1)*np.nan
        logging.info('Loaded {:d} episodes from {:s}'.format(
            tot, env_episodes_file))
        logging.info('x, xi_1toT, bads_1toT, rwd')
        logging.info(x.size()); logging.info(xi_1toT.size())
        logging.info(bads_1toT.size()); logging.info(rwd.size())
        return x, xi_1toT, bads_1toT, rwd

    @staticmethod
    def compute_xi_std(xi_1toT):
        # We do not standardize xi, since this would make scaling to a known
        # range problematic. Instead we report std for loss scaling.
        tot, seq_len, _ = xi_1toT.size()
        xi_std = xi_1toT.view(tot*seq_len, -1).std(dim=0)
        max_std = 1000.0; min_std = 1.0/max_std
        if (xi_std<min_std).any() or (xi_std>max_std).any():
            logging.warning('WARNING: extreme xi_std'); logging.warning(xi_std)
        xi_std = torch.clamp(xi_std, min_std, max_std).unsqueeze(0)
        logging.info('xi_1toT std'); logging.info(xi_std)
        return xi_std

    def get_xi_std(self):
        return self.xi_std

    def sample_good_bad(self):
        rnd_ids = torch.randperm(self.x.size(0))
        if rnd_ids.size(0)>1000: rnd_ids = rnd_ids[0:1000]  # look in 1K trajs
        badness = self.bads_1toT[rnd_ids].mean(dim=1)
        good_id = rnd_ids[badness.argmin()]
        bad_id = rnd_ids[badness.argmax()]
        maxrwd_id = rnd_ids[self.rwd[rnd_ids].argmax()]
        minrwd_id = rnd_ids[self.rwd[rnd_ids].argmin()]
        x = torch.stack([self.x[good_id], self.x[maxrwd_id],
                         self.x[bad_id], self.x[minrwd_id]], dim=0)
        xi_1toT = torch.stack(
            [self.xi_1toT[good_id], self.xi_1toT[maxrwd_id],
             self.xi_1toT[bad_id], self.xi_1toT[minrwd_id]], dim=0)
        bads_1toT = torch.stack(
            [self.bads_1toT[good_id], self.bads_1toT[maxrwd_id],
             self.bads_1toT[bad_id], self.bads_1toT[minrwd_id]], dim=0)
        rwd = torch.stack([self.rwd[good_id], self.rwd[maxrwd_id],
                           self.rwd[bad_id], self.rwd[minrwd_id]], dim=0)
        logging.info('sample_good_bad(): true_rwd'); logging.info(rwd)
        logging.info('true_goodness'); logging.info(1-bads_1toT.mean(dim=1))
        logging.info('x'); logging.info(x.size()); logging.info(x)
        return x, xi_1toT, bads_1toT, rwd


def load_checkpoint(checkpt_file, model, args, optimizer, device):
    # Note: model and optimizer should be defined in the calling code.
    checkpt_file = os.path.expanduser(checkpt_file)
    if not os.path.isfile(checkpt_file):
        logging.info('No checkpoint file'); logging.info(checkpt_file)
        assert(False)
    logging.info("=> loading checkpoint '{}'".format(checkpt_file))
    chkpt = torch.load(checkpt_file, map_location=device)
    epoch = chkpt['epoch']
    all_args = [chkpt['all_args'][0], args]  # args from all train restarts
    model.load_state_dict(chkpt['svae_dc_state_dict'])
    optimizer.load_state_dict(chkpt['svae_dc_optim_dict'])
    logging.info("Loaded chkpt '{}' (opt_step {})".format(checkpt_file, epoch))
    if device != 'cpu':
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return model, optimizer, epoch, all_args


def get_scaled_obs(obs, env):
    obs_ranges = env.observation_space.high - env.observation_space.low
    return (obs - env.observation_space.low) / obs_ranges


def make_env_from_args(env_name, seed):
    if env_name.startswith(('Yumi','Franka','Sawyer')):
        import gym_bullet_extensions # to register YumiPosition-v2 and rest
    if env_name.startswith('Daisy'):
        import gym_daisy_custom # to register DaisyCustom-v0 and rest
    env = gym.make(env_name); env.seed(seed)
    # Collect useful information about env specs.
    max_episode_steps = env.spec.tags[
        'wrapper_config.TimeLimit.max_episode_steps']
    print('Created env', env_name, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape,
          'max_episode_steps', max_episode_steps)
    return env, max_episode_steps


def make_policy_from_args(env, controller_class, max_episode_steps):
    env_name = env.spec.id
    if env_name.startswith(('Yumi', 'Franka', 'Sawyer')):
        from gym_bullet_extensions.control.waypts_policy import (
            WaypointsPosPolicy, WaypointsEEPolicy, WaypointsMinJerkPolicy,
            WaypointsVelPolicy)
    elif env_name.startswith('Daisy'):
        # DaisyGai11DPolicy for hardware, DaisyGai27DPolicy for sim,
        # DaisyTripodPolicy for debugging.
        from gym_daisy_custom.control.gaits import (
                DaisyGait27DPolicy, DaisyGait11DPolicy, DaisyTripod27DPolicy)
    else:
        print('Please import controller class for your env', env_name)
        assert(False)
    policy_kwargs = {'controller_class': eval(controller_class),
                     'controller_dim': eval(controller_class).DIM,
                     't_max': max_episode_steps, 'robot': env.robot}
    if hasattr(env, 'get_init_pos'):
        policy_kwargs['get_init_pos_fxn'] = env.get_init_pos
    policy = StructuredPolicy(**policy_kwargs)
    return policy
