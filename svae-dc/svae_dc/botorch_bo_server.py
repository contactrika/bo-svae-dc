#
# Bayesian Optimization server. Communicates over TCP sockets.
# Useful when running ROS nodes (what usually use older python versions).
#
# Note: botorch does maximization by default, so we will perform maximization
# of rewards directly (can be interpreted later as minimizing costs, if needed).
#
# @contactrika
#
import argparse
import logging
import os
import socket
import sys

import numpy as np
import torch
np.set_printoptions(precision=4, threshold=sys.maxsize, suppress=True)
torch.set_printoptions(precision=4, threshold=sys.maxsize, sci_mode=False)

from .utils.bo_constants import reward_lims
from .utils.bo_utils import (
    init_and_fit_gp, optimize_UCB, UpperConfidenceBound
)
from .utils.svae_dc_utils import (
    make_env_from_args, make_policy_from_args
)
from .svae_dc_load import svae_dc_load


class BotorchBOServer:
    def __init__(self, args):
        print('Loading BotorchBOServer...')
        self.args = args
        # Sanity check args.
        assert(args.bo_num_init >= 2)
        if args.bo_kernel_type == 'Random': assert(args.bo_num_trials == 0)
        # Set up logging both to terminal and log file.
        save_path = 'output_{:s}_{:s}_UCB{:0.1f}_run{:d}'.format(
            args.env_name, args.bo_kernel_type, args.bo_ucb_beta, args.run_id)
        #save_path += datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
        args.save_path = os.path.join(args.output_prefix, save_path)
        if not os.path.exists(args.save_path): os.makedirs(args.save_path)
        args.checkpt_path = os.path.join(args.save_path, 'checkpt-%04d.pth' % 0)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s",
            handlers=[logging.FileHandler(os.path.join(args.save_path, 'log.txt')),
                      logging.StreamHandler(sys.stdout)])
        logging.info(args)
        # Init randomness and GPUs.
        use_cuda = (args.gpu is not None) and torch.cuda.is_available()
        args.device = 'cuda:'+str(args.gpu) if use_cuda else 'cpu'
        np.random.seed(args.run_id); torch.manual_seed(args.run_id)
        if args.device != 'cpu':
            torch.cuda.set_device(args.gpu)
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.run_id)
        # Make env and policy from args.
        self.env, max_episode_steps = make_env_from_args(
            args.env_name, args.run_id)
        self.policy = make_policy_from_args(
            self.env, args.controller_class, max_episode_steps)
        # Load NNs.
        self.svae = None; self.svae_latent_dim = None
        load_only_adjustable = True
        if args.bo_kernel_type=='KL' or args.bo_kernel_type=='uKL':
            load_only_adjustable = False  # to get original trajectories
        if 'SVAE' in args.bo_kernel_type or args.bo_kernel_type.endswith('KL'):
            dyn_comp = True if '-DC' in args.bo_kernel_type else False
            self.svae, self.svae_latent_dim = svae_dc_load(
                self.env, self.policy, args.svae_dc_checkpt,
                args.svae_dc_override_good_th, args.env_name, args.device,
                dyn_comp, load_only_adjustable)
        self.gp = None  # caching for hyperparams
        self.online_adjust = args.online_adjust
        self.y_lims = reward_lims(self.args.env_name)
        self.run_id = args.run_id

    def bo_get_next_pt(self, X, Y):
        trial_num = 0 if (X is None or X.size==0) else X.shape[0]
        # Return random point when X is empty or has fewer than num_init pts.
        if trial_num < self.args.bo_num_init:
            logging.info('Random trial {:d}'.format(trial_num))
            next_x = np.random.rand(self.policy.params.shape[0])
            return next_x  # params in [0,1]; caller unscales params
        # Fit GP models with data obtained so far
        # As suggested in botorch/tutorials/closed_loop_botorch_only.ipynb
        # we re-initialize GP for each trial, but also reuse pytorch tensor
        # dictionary from previous iteration to speed up initialization.
        logging.info('================ BO {:s} trial {:d} ============='.format(
            self.args.bo_kernel_type, trial_num-self.args.bo_num_init))
        x_all = torch.from_numpy(X).float().to(self.args.device)
        y_all = torch.from_numpy(Y).float().to(self.args.device)
        x_size = x_all.size(-1)
        x_bounds = torch.stack([torch.zeros(x_size),
                                torch.ones(x_size)]).to(device=self.args.device)
        mll, gp = init_and_fit_gp(
            x_all, y_all, self.y_lims[0], self.y_lims[1],
            self.args.device, self.args.bo_kernel_type,
            self.svae, self.svae_latent_dim,
            self.gp, self.online_adjust, debug=self.args.debug)
        acq_fun = UpperConfidenceBound(gp, beta=self.args.bo_ucb_beta)
        next_x, best_acq_y = optimize_UCB(
            acq_fun, x_bounds, gp.unscale_y_fxn, seed=self.run_id)
        logging.info('best_acq_y {:0.4f} unscaled {:0.4f}'.format(
            best_acq_y, gp.unscale_y_fxn(best_acq_y)))
        x_all = torch.cat([x_all, next_x.unsqueeze(0)])
        gp.print_predict(x_all, y_all)
        # Return next_x point (scaled in [0,1], caller can unscale if needed).
        return next_x.detach().cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description="SVAE")
    # Overall args.
    parser.add_argument('--run_id', type=int, default=0,
                        help='Run id (also used as random seed)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id')
    parser.add_argument('--debug', type=int, default=0, help='Debug level')
    parser.add_argument('--svae_dc_checkpt', type=str, default=None)
    parser.add_argument('--svae_dc_override_good_th', type=float, default=None,
                        help='Threshold for traj to be considered acceptable')
    parser.add_argument('--online_adjust', type=int, default=0, choices=[0, 1],
                        help='Whether to adjust kernel NNs online')
    # BO args.
    parser.add_argument('--bo_num_init', type=int, default=2,
                        help='Number of initial random trials')
    parser.add_argument('--bo_kernel_type', type=str, default='SE',
                        help='BO kernel type',
                        choices=['Random', 'SE', 'Matern', 'uSE', 'uMatern',
                                 'SVAE-DC-SE', 'SVAE-DC-Matern',
                                 'SVAE-SE', 'SVAE-Matern',
                                 'KL', 'uKL', 'LKL'])
    parser.add_argument('--bo_ucb_beta', type=float, default=1.0,
                        help='Beta param for UCB')
    # Environment-related variables.
    parser.add_argument('--env_name', type=str, default='YumiVel-v2',
                        help='Gym env name string')
    parser.add_argument('--controller_class', type=str,
                        default='WaypointsVelPolicy',
                        choices=['DaisyGait11DPolicy', 'DaisyGait27DPolicy',
                                 'WaypointsPosPolicy', 'WaypointsMinJerkPolicy',
                                 'WaypointsEEPolicy', 'WaypointsVelPolicy'],
                        help='Controller class name')
    # Auxiliary args used when ran in batch mode (not used for bo_get_next_pt).
    parser.add_argument('--bo_num_trials', type=int, default=20,
                        help='Number of BO trials per run')
    parser.add_argument('--output_prefix', type=str,
                        default=os.path.expanduser('botorchdata/'))
    # Auxiliary when used in server mode (not used outside main_server())
    parser.add_argument('--port', type=int, default=10000,
                        help='Server TCP Port')
    args = parser.parse_args()
    return args


def maybe_show_waypts(bo):
    if (not hasattr(bo.policy, 'controller') or
        not hasattr(bo.policy.controller, 'waypts')): return  # nothing to do
    import pybullet
    if not hasattr(bo, 'tgt_viz_ids'):
        print('BO creating tgt_viz_ids')
        bo.tgt_viz_ids = []
        waypts = bo.policy.controller.waypts
        for i in range(waypts.shape[0]):
            alph = float(i+1)/waypts.shape[0]
            bo.tgt_viz_ids.append(bo.env.robot.create_visual_area(
                pybullet.GEOM_CYLINDER, [0,0,0],
                radius=0.03, rgba=[0,1,1,alph]))
    from env_demo import show_waypts
    show_waypts(bo.env, bo.policy, bo.tgt_viz_ids)


def run_bo_get_next_pt_server(bo, fname):
    data = np.load(fname)
    X = data['X']; Y = data['Y']
    print('read X', X, 'Y', Y)
    next_x = bo.bo_get_next_pt(X, Y)
    waypts = None; traj = None
    if hasattr(bo.policy, 'controller'):
        # Save the desired trajectory information, so that ros nodes do not have
        # to call python3 code (many ros distros still only ok with python2).
        bo.policy.set_params(torch.from_numpy(next_x))  # init controller
        bo.policy.print()
        if hasattr(bo.policy.controller, 'waypts'):
            waypts = bo.policy.controller.waypts
        if hasattr(bo.policy.controller, 'traj'):
            traj = bo.policy.controller.traj
    if 'Viz' in bo.env.spec.id:
        maybe_show_waypts(bo)
        from env_demo import play
        play(bo.env, bo.policy, num_episodes=1, num_randomized=0,
             resample_policy_params=False)
    # Save data for the client
    np.savez(fname, X=X, Y=Y, next_x=next_x, waypts=waypts, traj=traj)


def main_server():
    args = get_args()
    bo = BotorchBOServer(args)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # init TCP socket
    server_address = ('localhost', args.port)  # bind the socket to the port
    print('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    sock.listen(1)
    bufsize = 1024
    while True:
        print('waiting for a connection')
        connection, client_address = sock.accept()
        try:
            print('connection from', client_address)
            fname_bytes = connection.recv(bufsize)
            fname = fname_bytes.decode('utf-8')
            print('received "%s"' % fname)
            print('calling run_bo_get_next_pt')
            run_bo_get_next_pt_server(bo, fname)
            print('saved next_pt in file')
            connection.sendall(fname_bytes)
            print('done with client', client_address)
        finally:
            connection.close()  # clean up


if __name__ == "__main__":
    main_server()
