#
# Main for training Sequential VAE with Dynamic Compression (SVAE-DC).
#
# @contactrika
#
from datetime import datetime
import logging
import os
import sys
import time

import numpy as np
import torch
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)
#torch.set_printoptions(precision=2, linewidth=150, threshold=None)

from svae_dc import SVAE_DC
from svae_dc_args import get_all_args
from svae_dc_nets import NetsParams
from utils.visualize import visualize_samples
from utils.svae_dc_utils import load_checkpoint, SVAEDataset
from utils.svae_dc_utils import make_env_from_args, make_policy_from_args


def do_logging(epoch, elbo, svae_dc_params, optimizer, tb_writer, debug_dict):
    assert(len(optimizer.param_groups)==1)
    if tb_writer is not None:
        tb_writer.add_scalar('elbo', elbo.mean().item(), epoch)
        tb_writer.add_scalar(
            'optimizer_lr', optimizer.param_groups[0]['lr'], epoch)
        tb_writer.add_scalar(
            'logvar_limit',
            svae_dc_params.logvar_scheduler.logvar_limit(), epoch)
        for k,v in debug_dict.items():
            vv = v.mean().item() if type(v)==torch.Tensor else v
            tb_writer.add_scalar(k, vv, epoch)
    logging.info('Train epoch {:d} lr {:.2E} ELBO: {:.4f}'.format(
        epoch, optimizer.param_groups[0]['lr'], elbo.mean().item()))


def do_saving(args, svae_dc, optimizer, epoch, all_args, tb_writer):
    logging.info('Adding histograms to Tebsorboard')
    #for name,param in svae_dc.named_parameters():
    #    tb_writer.add_histogram(
    #        name,param.clone().cpu().data.numpy(), epoch)
    #for k,v in debug_hist_dict.items():
    #    tb_writer.add_histogram(
    #        k,v.clone().cpu().data.numpy(), epoch)
    if (epoch%(args.save_interval))==0:
        fbase = os.path.join(args.save_path,'checkpt-')
        oldf = fbase+'{:d}.pt'.format(epoch-2*args.save_interval)
        if os.path.exists(oldf): os.remove(oldf)
        checkpt_path = fbase+'{:d}.pt'.format(epoch)
        logging.info('Saving {:s}'.format(checkpt_path))
        torch.save({'svae_dc_state_dict':svae_dc.state_dict(),
                    'svae_dc_optim_dict':optimizer.state_dict(),
                    'all_args':all_args, 'epoch':epoch}, checkpt_path)
    logging.info('do_saving() done')


def learn_from_samples(x, xi_1toT, bads_1toT, rwd, xi_std, args,
                       svae_dc, svae_dc_params, optimizer, lr_scheduler, epoch,
                       env, policy, all_args, dataset, tb_writer, debug):
    svae_dc.train()  # set torch internal train flags
    optimizer.zero_grad()
    # Compute ELBO.
    obj, elbo, debug_dict, debug_delta = svae_dc.elbo(
        x, xi_1toT, bads_1toT, xi_std, use_laplace=args.svae_dc_use_laplace,
        gen_beta=args.svae_dc_gen_beta, debug=debug)
    if (obj!=obj).any(): raise ValueError('NaN in objective')
    # Compute gradients; the following line is the core of training.
    # Here we multiply the objective (ELBO) by -1 before backprop.
    obj.mean().mul(-1).backward()
    # Apply gradients; the following line is the core of training.
    optimizer.step()
    # Apply learning rate decay if scheduler is specified
    # Can use .step(elbo.mean().item()) for Plateau scheduler.
    if lr_scheduler is not None: lr_scheduler.step()
    svae_dc_params.logvar_scheduler.step(elbo.mean().item())
    if debug and debug_delta is not None:
        obs_names = env.get_obs_names()
        for nmid, nm in enumerate(obs_names):
            debug_dict[nm+'_delta'] = debug_delta[:,nmid]
        do_logging(epoch, elbo, svae_dc_params, optimizer,
                   tb_writer, debug_dict)


def train(args, svae_dc, svae_dc_params, optimizer, lr_scheduler, epoch,
          eval_env, policy, all_args, tb_writer):
    assert(args.env_episodes_file is not None)
    test_dataset = SVAEDataset(args.test_env_episodes_file)
    test_dloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=True,
        num_workers=8, pin_memory=True)
    # Main SVAE-DC training loop.
    # DataLoader has functionality to speed up CPU->GPU transfer
    # https://discuss.pytorch.org/t/pin-memory-vs-sending-direct-to-gpu-from-dataset/33891
    for data_pass in range(args.svae_dc_num_data_passes):
        logging.info('=========== Train pass {:d} =========='.format(data_pass))
        if data_pass%20==0:
            dataset = SVAEDataset(args.env_episodes_file)
            xi_std = dataset.get_xi_std().to(args.device, non_blocking=True)
            dloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=8, pin_memory=True)
        for x, xi_1toT, bads_1toT, rwd in dloader:
            # Train.
            x = x.to(args.device, non_blocking=True)
            xi_1toT = xi_1toT.to(args.device, non_blocking=True)
            bads_1toT = bads_1toT.to(args.device, non_blocking=True)
            rwd = rwd.to(args.device, non_blocking=True)
            learn_from_samples(
                x, xi_1toT, bads_1toT, rwd, xi_std, args,
                svae_dc, svae_dc_params, optimizer, lr_scheduler, epoch,
                eval_env, policy, all_args, dataset, tb_writer,
                (epoch%args.log_interval==0))
            # Test, viz, save.
            if epoch%args.save_interval==0:
                test_x, test_xi_1toT, test_bads_1toT, test_rwd = \
                    next(iter(test_dloader))
                test_dict = svae_dc.make_test_dict(
                    test_x.to(args.device), test_xi_1toT.to(args.device),
                    test_bads_1toT.to(args.device), test_rwd.to(args.device))
                for k,v in test_dict.items():
                    vv = v.mean().item() if type(v)==torch.Tensor else v
                    tb_writer.add_scalar(k, vv, epoch)
                gb_x, gb_xi_1toT, gb_bads_1toT, gb_rwd = \
                    test_dataset.sample_good_bad()
                visualize_samples(
                    svae_dc, gb_x.to(args.device), gb_xi_1toT.to(args.device),
                    gb_bads_1toT.to(args.device), gb_rwd.to(args.device),
                    args.save_path, args, epoch, eval_env, policy, tb_writer,
                    args.video_steps_interval)
                do_saving(args, svae_dc, optimizer, epoch, all_args, tb_writer)
            # Epoch done.
            epoch += 1
    logging.info('Training done!')


def main(args):
    # Set up logging both to terminal and log file.
    # Has to be done before tensorboard import.
    save_path = 'output_run' +  str(args.run_id) + '_'
    save_path += datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
    args.save_path = os.path.join(args.output_prefix, save_path)
    assert(not os.path.exists(args.save_path)); os.makedirs(args.save_path)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(args.save_path, 'log.txt')),
                  logging.StreamHandler(sys.stdout)])
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(args.save_path)  # setup tensorboard logging
    logging.info(args)

    # Init randomness and GPUs.
    # We don't need reproducibility for training, so favoring speed.
    # https://pytorch.org/docs/stable/notes/randomness.html
    use_cuda = (args.gpu is not None) and torch.cuda.is_available()
    args.device = 'cuda:'+str(args.gpu) if use_cuda else 'cpu'
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(args.seed)

    # Make eval env and policy from args.
    env, max_episode_steps = make_env_from_args(args.env_name, args.seed)
    if args.max_episode_steps is None:
        args.max_episode_steps = max_episode_steps  # use info from env spec
    policy = make_policy_from_args(
        env, args.controller_class, args.max_episode_steps)

    # Setup SVAE-DC.
    x_size = policy.get_params().shape[0]
    xi_size = env.observation_space.shape[0]
    if 'Daisy' in args.env_name:
        xi_dim_weights = torch.from_numpy(
            env.dim_weights).float().to(args.device)
        logging.info('Using xi_dim_weights'); logging.info(xi_dim_weights)
    else:
        xi_dim_weights = torch.ones(xi_size, device=args.device)
    T_xi = args.max_episode_steps+1
    K_tau = int(args.svae_dc_K_tau_scale*T_xi)
    svae_dc_params = NetsParams(
        x_size, xi_size, T_xi, xi_dim_weights,
        args.svae_dc_tau_size, K_tau, args.svae_dc_hidden_size,
        args.svae_dc_logvar_patience, args.debug)
    logging.info('SVAE_DC params:'); logging.info(svae_dc_params.__dict__)
    svae_dc = SVAE_DC(pr=svae_dc_params,
                      coder_type=args.svae_dc_coder_type,
                      latent_type=args.svae_dc_latent_type,
                      good_th=args.svae_dc_good_th,
                      dyn_comp=True,  # do learn dyn comp-related variables
                      device=args.device)

    # Setup optimizers.
    #optimizer = torch.optim.SGD(svae_dc.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.Adam(svae_dc.parameters(), lr=args.learning_rate,
    #                             amsgrad=True)
    optimizer = torch.optim.RMSprop(
        svae_dc.parameters(), lr=args.learning_rate, eps=1e-8, alpha=0.99)
    # See https://pytorch.org/docs/stable/optim.html
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='max', factor=0.99, patience=1000, verbose=True,
    #    threshold=5.0, threshold_mode='rel', cooldown=1000,
    #    min_lr=1e-6, eps=1e-08)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5000, 10000, 15000, 20000],
        gamma=0.5, last_epoch=-1)
    #lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
    #    optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=1000,
    #    step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None,
    #    scale_mode='cycle', last_epoch=-1)

    # If given, load and earlier checkpoint.
    if not args.svae_dc_load:
        epoch = 0; all_args = [args]
    else:  # if needed: load pre-trained model to continue training
        svae_dc, optimizer, epoch, all_args = load_checkpoint(
            args.svae_dc_load, svae_dc, args, optimizer, args.device)
        for i in range(epoch): lr_scheduler.step()  # record completed epochs

    # Train.
    train(args, svae_dc, svae_dc_params, optimizer, lr_scheduler, epoch,
          env, policy, all_args, tb_writer)

    # Cleanup.
    env.close()


if __name__ == '__main__':
    args = get_all_args()
    main(args)

