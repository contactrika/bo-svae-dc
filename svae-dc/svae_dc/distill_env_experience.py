#
# A quick script for equalizing the training data from env.
#
# This was useful for some versions of Daisy env+controllers that generated
# extremely unbalanced training data (robot rotating in place in most samples).
# Depending on our env you might need this in order to generate training data
# that VAE-based training can handle. VAEs are known to be prone to collapsing
# to the mean. A sequential VAE version helps a bit, since the trajectories
# could be somewhat more varied than individual samples. Though keep in mind
# that SVAE is not immune to the same problems that standard VAEs face.
#
# To launch multiple instances from shell run:
# for i in {10..19}; do `python distill_env_experience.py "experience/episodes10K_seed${i}_DaisyCustom-v0.npz" > /dev/null &`; done
#
from glob import glob
import os
import sys

import numpy as np
import torch
torch.set_printoptions(precision=2, linewidth=150, threshold=5000,
                       sci_mode=False, edgeitems=1000)

from utils.svae_dc_utils import make_env_from_args


def get_rwd_ranges(fname):
    if 'Daisy' in fname:
        return 5.0, None
    elif 'Yumi' in fname or 'Franka' in fname:
        return 0.2, None
    else:
        print('Episodes filename does not specify env type')
        assert('False')


def recompute_rwd_and_bads(env, xi_1toT_buf,
                           old_bads_1toT_buf, old_rwd_buf):
    rwd_buf = np.copy(old_rwd_buf)
    bads_1toT_buf = np.copy(old_bads_1toT_buf)
    print('Recomputing bads and rewards')
    obs_ranges = (env.observation_space.high-
                  env.observation_space.low)
    for bid in range(xi_1toT_buf.shape[0]):
        unscaled_obs = (xi_1toT_buf[bid,0,:]*obs_ranges+
                        env.observation_space.low)
        if 'Daisy' in env.spec.id:
            prev_state_dict = env.state_dict_from_state(unscaled_obs)
        bads_1toT_buf[bid,0,0] = False
        for st in range(1,xi_1toT_buf.shape[1]):
            unscaled_obs = (xi_1toT_buf[bid,st,:]*obs_ranges+
                            env.observation_space.low)
            if 'Daisy' in env.spec.id:
                state_dict = env.state_dict_from_state(unscaled_obs)
                bads_1toT_buf[bid,st,0] = env.get_is_bad(
                    prev_state_dict, state_dict)
                rwd_buf[bid] += env.get_reward_dense(
                    prev_state_dict, state_dict)
            else:
                env.override_state(unscaled_obs)
                bads_1toT_buf[bid,st,0] = env.get_is_bad()
            if 'Daisy' in env.spec.id: prev_state_dict = state_dict
        if (('Daisy' not in env.spec.id) and hasattr(env, 'final_rwd')):
            rwd_buf[bid] = env.final_rwd()
        if bid%100==0:
            msg = 'processed bid {:d} rwd {:0.4f} old_rwd {:0.4f}'
            msg += ' bads_frac {:0.4f} old bads_frac {:0.4f}'
            print(msg.format(bid, rwd_buf[bid][0], old_rwd_buf[bid][0],
                             bads_1toT_buf[bid].mean(),
                             old_bads_1toT_buf[bid].mean()))
    return rwd_buf, bads_1toT_buf


def main(files_pattern):
    equalize_reward_ranges = True
    fnames = glob(os.path.expanduser(files_pattern))
    assert(len(fnames)>0)  # make sure input files exist
    env_name = fnames[0].split('_')[-1].split('.')[0]
    print('Distill', env_name)
    assert(env_name in
           ['DaisyCustom-v0', 'YumiVel-v2', 'FrankaVel-v2', 'FrankaTorque-v2'])
    if env_name=='DaisyCustom-v0':
        env, max_episode_steps = make_env_from_args(env_name, 0)
    else:
        env = None  # turn off recomputation of reward and bads for manipulators
    for fname in fnames:
        print('Loading', fname)
        assert(os.path.exists(fname))
        data = np.load(fname)
        x_buf = data['x_buf']
        xi_1toT_buf = data['xi_1toT_buf']
        bads_1toT_buf = data['bads_1toT_buf']
        rwd_buf = data['rwd_buf']
        rnd_params_buf = None
        if 'rnd_params_buf' in data.keys():
            rnd_params_buf = data['rnd_params_buf']
        print('Loaded', rwd_buf.shape[0], 'points')
        #min_rwd = rwd_buf.min()
        #if ('Yumi' in fname or 'Franka' in fname) and (min_rwd < -2.0):
        #    min_rwd_id = rwd_buf.argmin()
        #    print('Invalid min reward {:0.4f} as id min_rwd_id'.format(
        #        min_rwd, min_rwd_id))
        #    ids = np.array([min_rwd_id,0])
        #    kwargs = {'x_buf':x_buf[ids], 'xi_1toT_buf':xi_1toT_buf[ids],
        #              'bads_1toT_buf':bads_1toT_buf[ids],
        #              'rwd_buf':rwd_buf[ids]}
        #    if rnd_params_buf is not None:
        #        kwargs['rnd_params_buf'] = rnd_params_buf[ids]
        #    np.savez('/tmp/x_bad.npz', **kwargs)
        #    assert(False)  # invalid min rwd
        if env is not None:
            rwd_buf, bads_1toT_buf = recompute_rwd_and_bads(
                env, xi_1toT_buf, bads_1toT_buf, rwd_buf)
        if equalize_reward_ranges:
            print('Equalizing reward ranges')
            good_rwd, bad_rwd = get_rwd_ranges(fname)
            ids_top = rwd_buf>=good_rwd
            num_good = ids_top.sum()
            ids_bottom = None
            if bad_rwd is not None:
                ids_bottom = rwd_buf<=bad_rwd
                if ids_top.shape[0]<ids_bottom.shape[0]:
                    ids_bottom = ids_bottom[:ids_top.shape[0]]
                ids_top_or_bottom = np.logical_or(ids_top, ids_bottom)
            else:
                ids_top_or_bottom = ids_top
            ids_middle = np.logical_not(ids_top_or_bottom)
            num_bads_seen = 0; cnt = 0
            while cnt<ids_middle.shape[0]:
                if ids_middle[cnt]==True: num_bads_seen += 1
                cnt += 1
                if num_bads_seen>=num_good: break
            ids_middle[cnt:] = False
            num_middle = ids_middle.sum()
            keep_ids = np.logical_or(ids_top_or_bottom, ids_middle).reshape(-1)
            print('Keeping num_good', num_good, 'num_middle', num_middle,
                  'total', keep_ids.sum())
            if bad_rwd is not None: print('ids_bottom', ids_bottom.sum())
            x_buf = x_buf[keep_ids]
            xi_1toT_buf = xi_1toT_buf[keep_ids,:,:]
            bads_1toT_buf = bads_1toT_buf[keep_ids,:,:]
            rwd_buf = rwd_buf[keep_ids,:]
        # save output file
        base_path = os.path.dirname(fname)
        basename = os.path.basename(fname)
        basename = 'distilled_'+basename
        outfl = os.path.join(base_path, basename)
        print('Saving', outfl)
        kwargs = {'x_buf':x_buf, 'xi_1toT_buf':xi_1toT_buf,
                  'bads_1toT_buf':bads_1toT_buf, 'rwd_buf':rwd_buf}
        if rnd_params_buf is not None: kwargs['rnd_params_buf'] = rnd_params_buf
        np.savez(outfl, **kwargs)


if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('Example usage: python postprocess_env_experience.py '+
              '\"/media/hd2tb/alldata/yumi_experience'+
              'episodes100K_seed0_YumiTorque-v2.npz\"')
        assert(False)
    files_pattern = os.path.expanduser(sys.argv[1])
    main(files_pattern)
