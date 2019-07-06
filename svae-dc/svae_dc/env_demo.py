#
# Demo for using env with policies.
#
import argparse
import logging
from glob import glob
import os
import sys
import time

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)
import torch
torch.set_printoptions(precision=4, linewidth=150, threshold=None, sci_mode=False)

from utils.svae_dc_utils import make_env_from_args, make_policy_from_args


def show_waypts(env, policy, tgt_viz_ids):
    robot = env.robot
    reset_pose = env.reset_qpos
    robot.reset_to_qpos(reset_pose)
    robot.refresh_viz()
    #input('Enter to show_waypts')
    ctrl = policy.controller
    robot.reset_to_qpos(reset_pose)
    for wpt in range(ctrl.waypts.shape[0]):
        ee_pos, ee_quat, fing_dist, _ = ctrl.parse_waypoint(ctrl.waypts[wpt])
        robot.sim.resetBasePositionAndOrientation(
            tgt_viz_ids[wpt], ee_pos, ee_quat)
        qpos = robot._ee_pos_to_qpos_raw(ee_pos, ee_quat, fing_dist)
        #qpos_ok = robot.ee_pos_to_qpos(ee_pos, ee_quat, fing_dist)
        robot.reset_to_qpos(qpos)
        robot.refresh_viz()
        input('Enter to show next waypt')
    robot.reset_to_qpos(reset_pose)
    robot.refresh_viz()
    input('Enter to continue')


def play(env, policy, num_episodes, num_randomized=0,
         scaled_policy_params=None, rnd_params=None,
         resample_policy_params=True):
    if scaled_policy_params is not None:
        assert(rnd_params is None or len(rnd_params)==0 or
               len(scaled_policy_params)==len(rnd_params))
    do_viz = ('Viz' in env.spec.id and
              env.spec.id.startswith(('Yumi', 'Franka', 'Sawyer')) and
              hasattr(policy, 'controller') and
              hasattr(policy.controller, 'waypts'))
    tgt_viz_ids = []
    if do_viz:
        import pybullet
        waypts = policy.controller.waypts
        for i in range(waypts.shape[0]):
            alph = float(i+1)/waypts.shape[0]
            tgt_viz_ids.append(env.robot.create_visual_area(
                pybullet.GEOM_CYLINDER, [0,0,0], radius=0.03, rgba=[0,1,1,alph]))
    for epsd in range(num_episodes):
        if scaled_policy_params is not None:
            params = policy.unscale_params(scaled_policy_params[epsd])
            policy.set_params(params); print('loaded policy')
            if rnd_params is not None and len(rnd_params)>0:
                assert(num_randomized==0)  # no need to call randomize()
                print('setting randomized params', rnd_params[epsd])
                env.set_randomize(rnd_params[epsd], debug=True)
        elif resample_policy_params:
            policy.resample_params(); print('resampled policy params')
        policy.print()
        if do_viz: show_waypts(env, policy, tgt_viz_ids)
        for rnd in range(max(1,num_randomized)):
            obs = env.reset()
            if num_randomized>0: env.randomize(debug=True)
            step = 0
            while True:
                action = policy.get_action(obs, t=step)
                next_obs, rwd, done, info = env.step(action)
                if done:
                    done_reward = info['done_reward']
                    done_badfrac = info['done_badfrac']
                    msg = 'Play epsd #{:d} done_reward {:0.4f} done_badfrac {:0.2f}'
                    logging.info(msg.format(epsd, done_reward, done_badfrac))
                    if done_badfrac<=0: input('Not bad! Press Enter to continue')
                    break
                obs = next_obs
                step += 1


def get_args():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description="EnvDemo")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--env_name', type=str,
                        default='DaisyCustomViz-v0', help='Env name')
    parser.add_argument('--controller_class', type=str,
                        default='DaisyGait11DPolicy',
                        choices=['DaisyGait11DPolicy', 'DaisyGait27DPolicy',
                                 'DaisyTripod27DPolicy',
                                 'WaypointsPosPolicy', 'WaypointsEEPolicy',
                                 'WaypointsMinJerkPolicy', 'WaypointsVelPolicy'],
                        help='Controller class name')
    parser.add_argument('--num_randomized', type=int, default=0,
                        help='Run same ctrl bu randomize env physic k times')
    parser.add_argument('--num_episodes', type=int, default=22,
                        help='Number of episodes')
    parser.add_argument('--policy_file', type=str, default=None,
                        help='numpy dump of policies to try')
    args = parser.parse_args()
    return args


def main(args):
    # Make env and policy from args.
    env, max_episode_steps = make_env_from_args(args.env_name, args.seed)
    print('max_episode_steps', max_episode_steps)
    policy = make_policy_from_args(
        env, args.controller_class, max_episode_steps)

    # Load external policy parameters if requested.
    scaled_policy_params_lst = None; rnd_params_lst = None
    if args.policy_file is not None:
        policy_files = sorted(glob(os.path.expanduser(args.policy_file)))
        assert(len(policy_files)>0)  # file pattern did not match any files
        scaled_policy_params_lst = []; rnd_params_lst = []
        for fl in policy_files:
            print('Loading', fl)
            data = np.load(fl)
            if 'x_all' in data.keys():
                x_all = data['x_all'][0:args.num_episodes]
                y_all = data['y_all'][0:args.num_episodes]
                rnd_params_all = None
                if 'rnd_params_all' in data.keys():
                    rnd_params_all = data['rnd_params_all']
                best_id = y_all.argmax()
                print('y_all', y_all.reshape(-1))
                print('x_all shape', x_all.shape, 'y_all shape', y_all.shape)
                if rnd_params_all is not None:
                    print('rnd_params_all', rnd_params_all.shape)
                if x_all.shape[0]>=10:
                    scaled_policy_params_lst.append(x_all[best_id,:])
                    if rnd_params_all is not None:
                        rnd_params_lst.append(rnd_params_all[best_id,:])
                    print('best_y', y_all[best_id], 'best_id', best_id)
                else:
                    for i in range(x_all.shape[0]):
                        scaled_policy_params_lst.append(x_all[i,:])
                        if rnd_params_all is not None:
                            rnd_params_lst.append(rnd_params_all[i,:])
            elif 'x_buf' in data.keys():
                x = data['x_buf']
                rwd = data['rwd_buf']
                rnd_params = None
                if 'rnd_params_buf' in data.keys():
                    rnd_params = data['rnd_params_buf']
                goodness = 1-data['bads_1toT_buf'].mean(axis=1)
                num_to_load = args.num_episodes  # num_episodes from each file
                if x.shape[0]<num_to_load: num_to_load = x.shape[0]
                for i in range(num_to_load):
                    scaled_policy_params_lst.append(x[i,:])
                    if rnd_params is not None:
                        rnd_params_lst.append(rnd_params[i,:])
                print('loaded {:d} policies from {:s}'.format(num_to_load, fl))
                print('rwd', rwd[0:num_to_load])
                print('goodness', goodness[0:num_to_load])
                print('rnd_params_lst', rnd_params_lst)
            else:
                print('x_all or x_buf in data should specify controllers')
                assert(False)
            assert(len(scaled_policy_params_lst)>0)  # check policies loaded
            if len(rnd_params_lst)>0:
                assert(len(scaled_policy_params_lst)==len(rnd_params_lst))
        args.num_episodes = len(scaled_policy_params_lst)
        print('scaled_policy_params_lst', scaled_policy_params_lst)

    # Play.
    play(env, policy, args.num_episodes, args.num_randomized,
         scaled_policy_params_lst, rnd_params_lst, resample_policy_params=True)

    # Cleanup.
    env.close()


if __name__ == "__main__":
    main(get_args())
