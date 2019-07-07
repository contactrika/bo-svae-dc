#
# SVAE-DC command-line arguments.
#
import argparse
import os


def get_all_args():
    parser = argparse.ArgumentParser(description="SVAE")
    # Overall args.
    parser.add_argument('--run_id', type=int, default=0, help='Run ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--debug', type=int, default=0, help='Debug level')
    parser.add_argument('--output_prefix', type=str,
                        default=os.path.expanduser('svaedata/'))
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every k train updates (epochs)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save every k train updates (epochs)')
    parser.add_argument('--video_steps_interval', type=int, default=10,
                        help='Interval for steps to sim and render for eval')
    # Flags for SVAE-DC
    parser.add_argument('--svae_dc_load', type=str, default='',
                        help='Checkpoint to load to continue training')
    parser.add_argument('--svae_dc_hidden_size', type=int, default=64,
                        help='Hidden size for all NNs')
    parser.add_argument('--svae_dc_tau_size', type=int, default=7,
                        help='Size of each tau_k (in tau_{1:K})')
    parser.add_argument('--svae_dc_K_tau_scale', type=float, default=0.05,
                        help='How much to shrink K=scale*T (T in xi_{1:T})')
    # We observed that when learning means and variances jointly, VAEs tend to
    # find extremely low-variance solutions. We experimented with limiting max
    # logvar at the early stages of the training (this was also used in prior
    # work we found as well). This produced ELBO curves that increase more
    # gradually, but ultimately this does not change the training outcome much.
    # So we kept the code, but turned off this param by default.
    parser.add_argument('--svae_dc_logvar_patience', type=int, default=None,
                        help='Number of epochs for logvar increase stages')
    parser.add_argument('--svae_dc_num_data_passes', type=int, default=10000,
                        help='Number of passes over training data')
    parser.add_argument('--svae_dc_use_laplace', type=int, default=1,
                        help='Whether to use Laplace instead of Gaussian')
    parser.add_argument('--svae_dc_coder_type', type=str, default='conv',
                        choices=['conv'],
                        help='Decoder/Encoder NN type')
    parser.add_argument('--svae_dc_latent_type', type=str, default='conv',
                        choices=['conv', 'mlp'],
                        help='NN type for latent dynamics')
    parser.add_argument('--svae_dc_good_th', type=float, default=0.35,
                        help='Threshold for traj to be considered acceptable')
    # This parameter could be used to specify custom weighting of gen vs recon
    # part of ELBO. We did not customize it in our work, and instead used a
    # default that simply adjusted for the different dimensionality of
    # original trajectories vs latent paths. Depending on your data you
    # could find it useful. Though a better alternative would be to equalize
    # the data using distill_env_experience. It generates balanced data
    # and can prevent VAE-based optimization from collapsing to the mean.
    parser.add_argument('--svae_dc_gen_beta', type=float, default=None,
                        help='How important to sync with gen/latent part.')
    # Environment-related variables.
    parser.add_argument('--env_name', type=str, default='YumiVel-v2',
                        help='Gym env name string')
    parser.add_argument('--controller_class', type=str,
                        default='WaypointsVelPolicy',
                        choices=['DaisyGait11DPolicy', 'DaisyGait27DPolicy',
                                 'DaisyTripod27DPolicy',
                                 'WaypointsPosPolicy', 'WaypointsMinJerkPolicy',
                                 'WaypointsEEPolicy', 'WaypointsVelPolicy'],
                        help='Controller class name')
    parser.add_argument('--max_episode_steps', type=int, default=None,
                        help='Length of env trajectories')
    parser.add_argument('--env_episodes_file', type=str, default=None,
                        help='Load episodes from file instead of collecting')
    parser.add_argument('--test_env_episodes_file', type=str, default=None,
                        help='Test trajectories (not used for training)')
    args = parser.parse_args()

    return args
