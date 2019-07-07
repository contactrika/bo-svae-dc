## BO-SVAE-DC code for CoRL2019 submission (anonymized)

### Installation

Set up virtualenv:
```
virtualenv --no-site-packages -p /usr/bin/python3.6 ENV
source ENV/bin/activate
pip install numpy
```
On OS X use: ```usr/local/bin/python3.6```, or specify another path where you have python3.6.

Install pybullet environments for Yumi and Franka:
```
cd gym-bullet-extensions
pip install -e .
cd ../
```

This installs various ABB Yumi and Franka Emika environments. The ones we used in our experiments are: YumiVel-v2, FrankaTorque-v2.
Viz suffix makes pybullet visualize. In the ending '-v2', 2 stands for the number of objects in the env.

Our code for the Daisy environment we used in our experiments is included in ```gym-daisy-custom``` module. However, the base class it uses for the pybullet Daisy simulator is in the process of being open-sourced by another group (should be out shortly). So all our work is included in this repo, but at the moment there is no way to give the public access to the lower-level pybullet Daisy simulation code.


Install SVAE-DC training and BO repo:
```
cd svae-dc
pip install -e .
cd ../
```

### Visualizing Yumi and Franka envs:
```
python svae-dc/svae_dc/env_demo.py \
 --env_name=FrankaTorqueViz-v2 \
 --controller_class=WaypointsMinJerkPolicy

python svae-dc/svae_dc/env_demo.py \
 --env_name=YumiVelViz-v2 \
 --controller_class=WaypointsVelPolicy
```
See args in ```svae-dc/svae_dc/env_demo.py``` for more options.

Note: Yumi visualization is quite compute-heavy for a machine without hardware graphics acceleration. It runs perfectly fine on a Linux Ubuntu desktop, but is quite slow on OS X Macbook Pro. On the other hand, Franka envs were fast on all machines we tested on.


### Data collection:

Example of collecting 1K episodes from FrankaTorque env:

```
python svae-dc/svae_dc/collect_env_experience.py \
 --env_name=FrankaTorque-v2 \
 --controller_class=WaypointsMinJerkPolicy \
 --output_prefix=experience/ \
 --num_episodes=1000 --seed=0 --num_procs=8
```
Note: OS X does not play well with multiprocessing options, so use ```--num_procs=1``` to turn off multiprocessing. If you are planning to train SVAE-DC, then you would likely want to collect 100K-1M trajectories. In that case you would likely be running this on a Unix system anyways. Here we are using 1K just as a quick example to show how to set up SVAE-DC training and BO. For Franka envs collecting 100K trajectories should take less than 1hr on a quad-core CPU machine, (the time depends on your exact compute specs, of course)

Run the above command with a different seed to get a shard of training data as well (e.g. ```--seed=1```)

### SVAE-DC training:

```
python svae-dc/svae_dc/svae_dc_main.py --gpu=0 \
 --env_name=FrankaTorque-v2 \
 --controller_class=WaypointsMinJerkPolicy \
 --learning_rate=1e-4 --batch_size=32 \
 --svae_dc_hidden_size=128 \
 --svae_dc_tau_size=4 --svae_dc_K_tau_scale=0.01 \
 --svae_dc_coder_type=conv \
 --svae_dc_latent_type=conv \
 --env_episodes_file=experience/episodes1K_seed0_FrankaTorque-v2.npz \
 --test_env_episodes_file=experience/episodes1K_seed1_FrankaTorque-v2.npz \
 --log_interval=100 --save_interval=1000 \
 --video_steps_interval=25 \
 --output_prefix=svaedata/
```
See ```svae-dc/svae_dc/svae_dc_args.py``` for more arguments.

All the SVAE-DC (NN) training is implemented in PyTorch. For visualization we use TensorboardX module to get a nice visual output for training. Start tensorboard:
```
tensorboard --logdir=svaedata/ --port=6006
```

Go to ```http://localhost:6006``` in you browser and you should see TensorBoard: 
* SCALARS tab will have plots
* IMAGES tab would have reconstruction and generation videos 
* TEXT tab will have parameters of the training job you launched

They should looks similar to this example image:

TODO: include tensorboard snapshot

After some time you should see recon (middle) and gen (right) roughly match the original (left). You can make smoother videos by setting a lower ```--video_steps_interval```, but that will take more CPU time.

Note on the training: 'bad frac' does not need to match between orig/recon/gen. As long as recon/gen have higher 'bad frac' that varies in the same direciton as the one printed in orig - that should be enough for BO. 

### Bayesian Optimization (BO):

SVAE-DC training will create checkpoints in directories that look like ```svaedata/output_run0_190707_120729/```. The last two checkpoint will be kept during training. Once you are satisfied with the training, pick a checkpoint and start BO:

```
python bo_main.py --gpu=0 --run_id=0 \
 --bo_num_init=2 --bo_num_trials=20 \
 --bo_kernel_type=SVAE-DC-SE \
 --env_name=FrankaTorque-v0 \
 --controller_class=DaisyGait11DPolicy \
 --svae_dc_override_good_th=0.3500 \
 --svae_dc_checkpt=/svaedata/output_run0_190707_120729/checkpt-70000.pt
```

```svae_dc_override_good_th``` parameter is not required, but can be useful. To see what a good value would be for your case: look at SVAE-DC training output logs (e.g. ```svaedata/output_run0_190707_120729/log.txt```). You will see printouts like this:

```
2019-07-07 12:13:04,124 Visualizing epoch 7
2019-07-07 12:13:04,227 knl res for bids 0 1
2019-07-07 12:13:04,233 raw_goodness
2019-07-07 12:13:04,233 [0.5116 0.5116]
2019-07-07 12:13:04,235 goodness
2019-07-07 12:13:04,236 [0.9827 0.9827]
...
2019-07-07 12:13:04,280 knl res for bids 2 3
2019-07-07 12:13:04,286 raw_goodness
2019-07-07 12:13:04,287 [0.5116 0.5116]
2019-07-07 12:13:04,288 goodness
2019-07-07 12:13:04,288 [0.9827 0.9827]
2019-07-07 12:13:04,289 raw distance (norm) 0.0016
```

TODO 

<hr />

## Notes

The above installation and demo instruction have been tested on Linux Ubuntu 18.04 and OS X 10.14.5 (Mojave). Key version numbers (that pip served at the time we wrote instructions):
```
numpy-1.16.4
scipy==1.3.0
six==1.12.0
pybullet-2.5.1
moviepy==1.0.0
mpi4py==3.0.2
torch==1.1.0.post2
tensorflow==1.14.0
tensorboard==1.14.0
tensorboardX==1.8
gpytorch==0.3.3
botorch==0.1.1
```


<hr />

To make sure numpy is enabled for PyBullet try:
```
python
>import pybullet
>pybullet.isNumpyEnabled()
```

If you get 0 as the result of the above then try:
```
pip uninstall pybullet
pip --no-cache-dir install pybullet
```

If pybullet simulator fails to install on your machine, sometimes it is easier to install it from source. For installation troubleshooting and further information on PyBullet see https://pybullet.org

<hr />

BoTorch code from Facebook AI is still experimental and rapidly developing. You will notice comments about it in the code. If you are running on a GPU with a limited memory: reduce the number of internal samples used during BO optimization: see ```svae-dc/svae_dc/utuils/bo_constants.py```

<hr />
