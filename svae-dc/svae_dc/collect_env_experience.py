#
# A quick script for collecting experience from env.
#
import argparse
import multiprocessing
import pickle
import os

import numpy as np

from utils.svae_dc_utils import (
    make_env_from_args, make_policy_from_args, get_scaled_obs
)

def get_args():
    parser = argparse.ArgumentParser(description="CollectEnvExperience")
    parser.add_argument('--env_name', type=str, default='YumiVel-v2',
                        help='Gym env name string')
    parser.add_argument('--controller_class', type=str,
                        default='WaypointsVelPolicy',
                        choices=['DaisyGait11DPolicy', 'DaisyGait27DPolicy',
                                 'DaisyTripod27DPolicy',
                                 'WaypointsPosPolicy', 'WaypointsMinJerkPolicy',
                                 'WaypointsEEPolicy', 'WaypointsVelPolicy'],
                        help='Controller class name')
    parser.add_argument('--ptraj_pkl_file', type=str, default=None,
                        help='File with planned trajectories to simulate')
    parser.add_argument('--output_prefix', type=str,
                        default=os.path.expanduser('~/local/experience/'))
    parser.add_argument('--num_procs', type=int, default=8,
                        help='Number of processes for envs')
    parser.add_argument('--num_episodes', type=int, default=100000,
                        help='Number of env processes for traj collection')
    parser.add_argument('--norandomize', action='store_true',
                        help='Turn off randomize of simulator physics params')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    return args


def init_from_moveit_ptraj(policy, ptraj, env):
    ptraj_r_pos = ptraj[2]  # get qpos
    ptraj_r_qvel = ptraj[3]  # TODO: remove hack with testing qvels
    policy.controller.traj[:,:] *= 0  # zero out all velocities
    tr_st = 0
    if ptraj_r_qvel is not None:
        sim_st_per_plan_st = int(policy.controller.traj.shape[0]/len(ptraj_r_qvel))
        print('policy.controller.traj.shape[0]', policy.controller.traj.shape[0])
        print('sim_st_per_plan_st', sim_st_per_plan_st)
        print('len(ptraj_r_vel)', len(ptraj_r_qvel))
        pl_tr_st = 0
        while pl_tr_st < len(ptraj_r_qvel):  # override right arm qpos
            print('pl_tr_st', pl_tr_st)
            print('ptraj_r_qvel[pl_tr_st]', ptraj_r_qvel[pl_tr_st])
            for tr_jid in range(len(ptraj_r_qvel[pl_tr_st])):
                policy.controller.traj[tr_st,tr_jid] = \
                    ptraj_r_qvel[pl_tr_st][tr_jid]
            tr_st += 1; pl_tr_st +=1
            for sub_tr_st in range(sim_st_per_plan_st-1):
                policy.controller.traj[tr_st,:] = \
                    policy.controller.traj[tr_st-1,:]
                tr_st += 1
    print('final tr_st', tr_st)
    while tr_st+1<policy.controller.traj.shape[0]:
        policy.controller.traj[tr_st+1,:] = \
            policy.controller.traj[tr_st,:]  # use prev qpos
        tr_st += 1
    policy.controller.traj[:,-2] = env.robot.kp
    policy.controller.traj[:,-1] = env.robot.kd

def collect_episodes(args):
    np.random.seed(args.seed)
    # Make env and policy from args.
    env, args.max_episode_steps = make_env_from_args(args.env_name, args.seed)
    policy = make_policy_from_args(env, args.controller_class,
                                   args.max_episode_steps)
    ptraj_data = None
    if args.ptraj_pkl_file is not None:
        assert(env.robot.control_mode == 'velocity')
        assert(args.controller_class == 'WaypointsPolicy')
        ptraj_data = pickle.load(
            open(os.path.expanduser(args.ptraj_pkl_file), 'rb'))
        args.num_episodes = len(ptraj_data)
    # Set up buffers.
    x_size = policy.get_params().shape[0]
    xi_size = env.observation_space.shape[0]
    seq_len = args.max_episode_steps+1
    x_buf = np.zeros([args.num_episodes, x_size])*np.nan
    xi_1toT_buf = np.zeros([args.num_episodes, seq_len, xi_size])
    bads_1toT_buf = np.zeros([args.num_episodes, seq_len, 1])
    rwd_buf = np.zeros([args.num_episodes, 1])
    rnd_params_buf = None
    if not args.norandomize:
        rnd_size = env.randomize().shape[0]
        rnd_params_buf = np.zeros([args.num_episodes, rnd_size])
    # Collect episodes from env.
    for sid in range(args.num_episodes):
        if sid%10==0: print('epsd {:d}'.format(sid))
        if ptraj_data is None:
            policy.resample_params()  # sample new random policy params
        else:
            init_from_moveit_ptraj(policy, ptraj_data[sid], env)
        unscaled_policy_params = policy.get_params()
        scaled_policy_params = policy.scale_params(unscaled_policy_params)
        policy.check_scaled_params(scaled_policy_params)
        x_buf[sid,:] = scaled_policy_params
        step = 0
        obs = env.reset()
        if not args.norandomize: rnd_params_buf[sid,:] = env.randomize()
        xi_1toT_buf[sid,step,:] = get_scaled_obs(obs, env)
        bads_1toT_buf[sid,step] = 0  # assume start state is not bad
        while True:
            env.render()
            action = policy.get_action(obs, t=step)
            obs, rwd, done, info = env.step(action)
            step += 1  # increment step before recording obs
            xi_1toT_buf[sid,step,:] = get_scaled_obs(obs, env)
            bads_1toT_buf[sid,step] = float(info['is_bad'])
            if done:
                xi_1toT_buf[sid,step,:] = get_scaled_obs(info['done_obs'], env)
                rwd_buf[sid] = info['done_reward']
                break
    return x_buf, xi_1toT_buf, bads_1toT_buf, rwd_buf, rnd_params_buf


def main_multiproc(args):
    # Process args and create args list for workers.
    episodes_per_proc = int(args.num_episodes/args.num_procs+0.5)  # ceil
    pool = multiprocessing.Pool(processes=args.num_procs)
    args_for_runs = []
    for rn in range(args.num_procs):
        rn_args = get_args(); rn_args.seed = args.seed*10000+rn
        rn_args.num_episodes = episodes_per_proc
        args_for_runs.append(rn_args)
    # Run worker pool and combine results.
    print('args_for_runs', args_for_runs)
    mp_results = pool.map_async(collect_episodes, args_for_runs)
    mp_results.wait()
    results = mp_results.get()
    x_buf, xi_1toT_buf, bads_1toT_buf, rwd_buf, rnd_params_buf = zip(*results)
    x_buf = np.vstack(x_buf)
    xi_1toT_buf = np.vstack(xi_1toT_buf)
    bads_1toT_buf = np.vstack(bads_1toT_buf)
    rwd_buf = np.vstack(rwd_buf)
    rnd_params_buf = np.vstack(rnd_params_buf)
    return x_buf, xi_1toT_buf, bads_1toT_buf, rwd_buf, rnd_params_buf


def main(args):
    if args.ptraj_pkl_file is not None: assert(args.num_procs==1)
    x_buf, xi_1toT_buf, bads_1toT_buf, rwd_buf, rnd_params_buf = \
        collect_episodes(args) if args.num_procs<=1 else main_multiproc(args)
    # Save to disk.
    save_path = args.output_prefix
    if not os.path.exists(save_path): os.makedirs(save_path)
    outfl = os.path.join(save_path, 'episodes{:d}K_seed{:d}_{:s}.npz'.format(
        int(args.num_episodes/1000), args.seed, args.env_name))
    kwargs = {'x_buf':x_buf, 'xi_1toT_buf':xi_1toT_buf,
              'bads_1toT_buf':bads_1toT_buf, 'rwd_buf':rwd_buf}
    if rnd_params_buf is not None: kwargs['rnd_params_buf'] = rnd_params_buf
    np.savez(outfl, **kwargs)
    if args.num_episodes<20:
        print('rwd_buf', rwd_buf)
        print('goodness', 1-bads_1toT_buf.mean(axis=1))
        print('rnd_params_buf', rnd_params_buf)


if __name__ == '__main__':
    main(get_args())

