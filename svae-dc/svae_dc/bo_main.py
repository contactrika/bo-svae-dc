#
# Main script for launching BO experiments with SVAE-DC kernel.
#
import logging
import shutil
import os
import sys

import numpy as np
import torch
np.set_printoptions(precision=4, threshold=sys.maxsize, suppress=True)
torch.set_printoptions(precision=4, threshold=sys.maxsize, sci_mode=False)

from botorch_bo_server import BotorchBOServer, get_args


def objective_fxn(env, policy, y_lims):
    step = 0
    obs = env.reset()
    while True:
        action = policy.get_action(obs, t=step)
        obs, rwd, done, info = env.step(action)
        step += 1
        if done:
            badfrac = 1.0*info['done_badfrac']
            reward = 1.0*info['done_reward']
            break
    logging.info('raw reward {:0.4f} badfrac {:0.4f}'.format(reward, badfrac))
    if reward<y_lims[0]: reward = y_lims[0]
    if reward>y_lims[1]: reward = y_lims[1]
    return reward


def run_bo(args):
    bo = BotorchBOServer(args)
    bo.env.randomize(debug=True)
    x_all = np.array([]); y_all = np.array([])
    for trial_id in range(args.bo_num_init+args.bo_num_trials):
        next_x = bo.bo_get_next_pt(x_all, y_all)
        # Evaluate next_x in the environment.
        unscaled_x = bo.policy.unscale_params(next_x)
        bo.policy.set_params(unscaled_x)
        next_y_val = objective_fxn(bo.env, bo.policy, bo.y_lims)
        if x_all.size==0:
            x_all = np.expand_dims(next_x, axis=0)
            y_all = np.array([[next_y_val]])
        else:
            x_all = np.vstack([x_all, next_x])
            y_all = np.vstack([y_all, np.array(next_y_val)])
        msg = '<------------------------------ max rwd {:0.4f} curr rwd {:0.4f}'
        logging.info(msg.format(y_all.max(), next_y_val))
        logging.info('curr x'); logging.info(next_x)
        if (trial_id+1)>=20 and (trial_id+1)%5==0:  # save results in a file
            outfl = os.path.join(args.save_path,
                                 'x_y_all_run'+str(args.run_id)+'.npz')
            if os.path.exists(outfl): shutil.move(outfl, outfl+'.old')
            np.savez(outfl, x_all=x_all, y_all=y_all)
    # Cleanup.
    bo.env.close()


def main():
    args = get_args()
    run_bo(args)


if __name__ == '__main__':
    main()
