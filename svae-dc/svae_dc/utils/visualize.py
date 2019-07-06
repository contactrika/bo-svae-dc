#
# Code to visualize samples and latent path for SVAE-DC.
#
import logging
import os
import time

import numpy as np
import torch

from .bo_kernels import KernelTransform


VIZ_BLUE = (0,0,255)
IMAGE_WIDTH = 300


def annotate(name, bid, step, np_img, the_bads,
             x=None, the_rwd=None, width=None):
    from PIL import Image, ImageFont, ImageDraw
    img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(img)
    fnt_path = os.path.join(os.path.split(__file__)[0], 'FreeMono.ttf')
    fnt = ImageFont.truetype(fnt_path, 15)
    fntsm = ImageFont.truetype(fnt_path, 8)
    msg = '{:s} {:d}\nstep {:d}'.format(name, bid, step)
    if the_bads is not None: msg += '\nbads frac {:0.2f}'.format(the_bads)
    if the_rwd is not None: msg += '\nrwd {:0.6f}'.format(the_rwd)
    draw.text((0,0), msg, font=fnt, fill=VIZ_BLUE)
    if x is not None and x.shape[0] < 30:
        msg = 'x='
        msg += np.array2string(
            x, formatter={'float_kind':lambda x: "%.2f\n" % x})
        draw.text((int(0.85*width),0), msg, font=fntsm, fill=VIZ_BLUE)
    return np.array(img)


def combine(all_imgs, num_steps, w, h):
    from PIL import Image
    combo_imgs = []
    for st in range(num_steps):
        combo_img = Image.new('RGBA', (w*len(all_imgs.keys()), h))
        x_offset = 0
        for name in all_imgs.keys():
            im = all_imgs[name][st]
            combo_img.paste(Image.fromarray(im), box=(x_offset,0))
            x_offset += w
        combo_imgs.append(np.array(combo_img))
    return combo_imgs


def add_video(name, imgs, epoch, tb_writer):
    strt = time.time()
    from tensorboardX import summary
    from tensorboardX.proto.summary_pb2 import Summary
    vid_imgs = np.array(imgs)  # TxHxWxC, C=RGBA
    video = summary.make_video(vid_imgs, fps=24)
    vs = Summary(value=[Summary.Value(tag=name+'_video', image=video)])
    tb_writer.file_writer.add_summary(vs, global_step=epoch)
    # add_video takes NxTxHxWxC and fails on RGBA
    #tb_writer.add_video(name+'_video', vid_tensor=vid_imgs, fps=4)
    print('{:s} video took {:2f}'.format(name, time.time()-strt))


def simulate(sim_env, policy, data, bid, xi_seq_len, video_steps_interval):
    all_imgs = {}
    strt = time.time()
    # Simulate frames.
    for name in ['orig', 'recon', 'gen']:
        unscaled_xis = data[name+'_unscaled_xi_1toT'][bid]
        all_imgs[name] = []
        the_bads = data[name+'_bads'][bid,0]
        # Visualize frames with overriding simulation state.
        sim_env.reset()
        for st in range(0,xi_seq_len,video_steps_interval):
            sim_env.override_state(unscaled_xis[st])
            env_img = sim_env.render_debug(width=IMAGE_WIDTH)
            if name == 'orig':
                np_img = annotate(
                    name, bid, st, env_img,
                    the_bads, data['unscaled_x'][bid],
                    data['orig_rwd'][bid,0], IMAGE_WIDTH)
            else:
                np_img = annotate(name, bid, st, env_img, the_bads)
            all_imgs[name].append(np_img)
    print('viz traj {:d} took {:2f}'.format(bid, time.time()-strt))
    num_frames = len(all_imgs['orig'])
    combo_imgs = combine(all_imgs, num_frames, IMAGE_WIDTH, IMAGE_WIDTH)
    return combo_imgs


def visualize_samples(model, x, orig_xi_1toT, orig_bads_1toT, rwd,
                      outdir, args, epoch, env, policy,
                      tb_writer=None, video_steps_interval=1):
    logging.info('Visualizing epoch {:d}'.format(epoch))
    model.eval()  # set internal eval flags
    if epoch==0:  # print command-line arg values for easy inspection
        args_str = ''
        for arg in vars(args):
            # Tensorboard uses markdown-like formatting, hence '  \n'.
            args_str += '  \n{:s}={:s}'.format(
                str(arg), str(getattr(args, arg)))
        tb_writer.add_text('args', args_str, epoch)
    assert ((type(orig_xi_1toT) == torch.Tensor) and (orig_xi_1toT.dim() == 3))
    batch_size, xi_seq_len, xi_size = orig_xi_1toT.size()
    data = {}
    # Generate.
    gen_xi_1toT_distr, gen_bads_1toT_distr, _ = model.generate(x)
    gen_xi_1toT = gen_xi_1toT_distr.mu.view(batch_size, xi_seq_len, -1)
    # Reconstruct.
    recon_xi_1toT_distr, recon_bads_1toT_distr, recon_tau_1toK, _ = \
        model.reconstruct(orig_xi_1toT, orig_bads_1toT,
                          require_grad=False, debug=False)
    recon_xi_1toT = recon_xi_1toT_distr.mu.view(batch_size, xi_seq_len, -1)
    # Print tau information.
    knl = KernelTransform(model)
    for bid in range(x.size(0)):
        for other_bid in range(bid+1,x.size(0)):
            logging.info('knl res for bids {:d} {:d}'.format(bid, other_bid))
            knl_res, _ = knl.apply(
                torch.stack([x[bid,:], x[other_bid,:]]), debug=True)
            diff = knl_res[0]-knl_res[1]
            logging.info('raw distance (norm) {:0.4f}'.format(
                torch.norm(diff).squeeze().detach().cpu().numpy()))
    # Prepare original data.
    data['orig_bads'] = orig_bads_1toT.mean(dim=1)
    data['orig_rwd'] = rwd
    # Unscale trajectories and params.
    obs_low = torch.from_numpy(env.observation_space.low).unsqueeze(0).to(
        args.device)
    obs_high = torch.from_numpy(env.observation_space.high).unsqueeze(0).to(
        args.device)
    obs_ranges = obs_high - obs_low
    clp_gen_xi_1toT = torch.clamp(gen_xi_1toT, 0, 1)
    data['gen_unscaled_xi_1toT'] = clp_gen_xi_1toT*obs_ranges+obs_low
    data['gen_bads'] = gen_bads_1toT_distr.mu.view(
        batch_size, -1, 1).mean(dim=1)  # assuming y_size==1
    clp_recon_xi_1toT = torch.clamp(recon_xi_1toT, 0, 1)
    data['recon_unscaled_xi_1toT'] = clp_recon_xi_1toT*obs_ranges+obs_low
    data['recon_bads'] = recon_bads_1toT_distr.mu.view(
        batch_size, -1, 1).mean(dim=1)  # assuming y_size==1
    clp_orig_xi_1toT = torch.clamp(orig_xi_1toT, 0, 1)
    data['orig_unscaled_xi_1toT'] = clp_orig_xi_1toT*obs_ranges+obs_low
    unscaled_x = policy.unscale_params(x)
    policy.check_params(unscaled_x)
    data['unscaled_x'] = unscaled_x
    for k,v in data.items(): data[k] = v.clone().cpu().data.numpy()
    # Output data stats to Tensorboard.
    #if tb_writer is not None:
    #    for k,v in data.items():
    #        tb_writer.add_histogram(k, v, epoch)
    # Get simulation env, if possible.
    # Simulate to compare (using unscaled policy (x) and trajectories (xi)).
    if tb_writer is not None and env is not None:
        all_combo_imgs = []
        for bid in range(x.size(0)):
            scaled_params = x[bid].detach().cpu()
            policy.set_params(policy.unscale_params(scaled_params))
            combo_imgs = simulate(env, policy, data, bid,
                                  xi_seq_len, video_steps_interval)
            all_combo_imgs.extend(combo_imgs)
        add_video('orig-recon-gen', all_combo_imgs, epoch, tb_writer)
    else:
        np.savez(os.path.join(outdir,
                              'train_viz_{:05d}.npz'.format(epoch)), data)
