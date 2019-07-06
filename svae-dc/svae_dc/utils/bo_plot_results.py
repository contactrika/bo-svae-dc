#
# A quick script to plot BO results.
#
import os

import numpy as np

from scipy import stats
import matplotlib.pyplot as plt

NAMES = {'Random':'Random', 'SE':'SE', 'uSE':'SE',
         'Matern':'Matern', 'uMatern':'Matern',
         'SVAE-DC-SE':'SVAE-DC',  'SVAE-DC-Matern':'SVAE-DC-Matern',
         'SVAE-SE':'SVAE',  'SVAE-Matern':'SVAE-Matern',
         'KL':'BBK-KL', 'uKL':'BBK-KL'}
MARKERS = {'Random':'3', 'SE':'x', 'uSE':'x', 'Matern':'s', 'uMatern':'s',
           'SVAE-DC-SE':'P',  'SVAE-DC-Matern':'^',
           'SVAE-SE':'+',  'SVAE-Matern':'$\Delta$',
           'KL':'.', 'uKL':'.'}
COLORS = {'Random':'k', 'SE':'#A0522D', 'uSE':'#A0522D',
          'Matern':'#FF7F50	', 'uMatern':'#FF7F50',
          'SVAE-DC-SE':'b',  'SVAE-DC-Matern':'b',
           'SVAE-SE':'y',  'SVAE-Matern':'y',
           'KL':'#ff1493', 'uKL':'#ffc0cb' }


def plot_all_algos(env_name, algos, y_lims, prefix, baselines_prefix=None,
                   nruns=50, ntrials=22, ninit=2,
                   names=None, markers=None, colors=None, debug=False):
    if names is None: names = NAMES
    if markers is None: markers = MARKERS
    if colors is None: colors = COLORS
    if baselines_prefix is None: baselines_prefix = prefix
    hs = []; labels = []
    fig = plt.figure(figsize=(10, 10))
    for algo_id, algo in enumerate(algos):
        # Put together data from 10 runs.
        not_baseline = ('SVAE' in algo or 'KL' in algo)
        prfx = os.path.expanduser(prefix if not_baseline else baselines_prefix)
        y_all_runs = []
        rn = -1
        while len(y_all_runs) < nruns:
            rn += 1; assert(rn<100)
            algo_str = 'output_'+env_name+'_'+algo+'_UCB1.0_run'+str(rn)
            fname = os.path.join(prfx, algo_str, 'x_y_all_run'+str(rn)+'.npz')
            print('looking for ', fname)
            if not os.path.exists(fname): continue
            print('loading', fname)
            data = np.load(fname)
            y_all_runs.append(data['y_all'][0:ntrials].squeeze())
        y_all_runs = np.vstack(y_all_runs)
        # Plot.
        labels.append(names[algo])
        hs.append(plot_algo(y_all_runs, marker=markers[algo],
                  color=colors[algo], label=names[algo], ylims=y_lims))
        # Informative x axis labels
        xticks = []; xlbls = []
        for i in range(ninit):
            if ntrials<20 or i>0:
                xticks.append(i)
                xlbls.append('init\n'+str(i+1))
        for i in range(y_all_runs.shape[1]):
            if ntrials<20 or (i+1)%5==0:
                xticks.append(i+ninit); xlbls.append(str(i+1))
        plt.xticks(xticks, xlbls, rotation=0, fontsize=25)
        if debug: print(algo, 'y_all_runs', y_all_runs)
    # Annotate and save figure.
    #plt.title('BO '+env_name, fontsize=35)
    plt.xlabel('trial', fontsize=25, labelpad=-20)
    plt.ylabel('best reward so far', fontsize=30) # , labelpad=-20)  # for Yumi
    loc = 'upper left' if ntrials<=20 else 'lower right'
    plt.legend(hs, labels, loc=loc, fontsize=20)
    plt.tight_layout()
    fig.savefig('bo_plot_'+env_name+'.png')
    plt.close(fig)


def plot_algo(y_all_runs, marker='^',  color='k', label='', ylims=None):
    num_runs, num_trials = y_all_runs.shape
    # Process.
    Y = np.zeros([num_runs, num_trials])*np.nan
    for rn in range(num_runs):
        for trial_id in range(num_trials):
            Y[rn, trial_id] = y_all_runs[rn,0:trial_id+1].max()
    # Plot.
    y_mean = np.nanmean(Y, axis=0)
    y_err = np.nanstd(Y, axis=0)
    msz = 12 if (marker == 'v') else 9 if (marker == 's') else 14
    linestyle = 'None' if num_trials>=20 else '--'
    for trial in range(num_trials):
        y_err[trial] = get_half_of_ci_size(Y[:,trial])
    elinewidth = 1; capsize = None
    if y_all_runs.shape[1]<20:  # turn on if CIs too large to see
        msz += 6
        if label=='Random':
            #elinewidth = 0.2; capsize = 2  # for Yumi hw
            elinewidth = 0.2; capsize = 2
        elif label=='SE':
            elinewidth = 0.4; capsize = None
        else:
            elinewidth = 1.5; capsize = 4
    x = range(len(y_mean))
    h = plt.errorbar(x, y_mean, y_err, errorevery=1,
                     label=label, color=color,
                     linestyle=linestyle, linewidth=2,
                     marker=marker, markersize=msz,
                     elinewidth=elinewidth, capsize=capsize)
    plt.xlim([0,num_trials])
    plt.tick_params(labelsize=30)
    if ylims is not None: plt.ylim(ylims)
    return h


def get_half_of_ci_size(vals, confidence_level=0.95):
    #confidence_level should be: p=0.975 for CI of 95% (0.95 for 90%)
    n = len(vals)
    sigma = np.nanstd(vals)  # one standard deviation
    alpha = 1 - confidence_level
    # using alpha/2 because we aim to return get_half_of_ci_size()
    # errorbar function will 'paste' each half around the mean
    t_mult = stats.t.ppf(1-alpha/2, n-1)
    ci = t_mult*sigma/np.sqrt(n)
    return ci


if __name__ == '__main__':
    env_name = 'DaisyCustomLong-v0'
    prefix = '/tmp/'+env_name+'_bo_runs'
    algos = ['Random', 'uMatern', 'SVAE-Matern', 'SVAE-DC-Matern']
    y_lims = [-5.0, 30.0] if 'Daisy' in env_name else [0, 0.7]
    if 'Franka' in env_name: y_lims = [-0.7, 0.7]
    # Now plot.
    plot_all_algos(env_name, algos, y_lims, prefix)
