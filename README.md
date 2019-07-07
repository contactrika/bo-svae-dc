## BO-SVAE-DC code for CoRL2019 submission (anonymized)

### Installation

Set up virtualenv:
```
virtualenv --no-site-packages -p /usr/local/bin/python3.6 ENV
source ENV/bin/activate
```

Install pybullet environments for Yumi and Franka:
```
cd gym-bullet-extensions
pip install -e .
cd ../
```

This installs various ABB Yumi and Franka Emika environments. The ones we used in our experiments are: YumiVel-v2, FrankaTorque-v2.
Viz suffix makes pybullet visualize. In the ending '-v2', 2 stands for the number of objects in the env.

If pybullet simulator fails to install on your machine, sometimes it is easier to install it from source. For installation troubleshooting and further information on PyBullet see https://pybullet.org

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

Note: Yumi visualization is quite compute-heav for  a machine without hardware graphics acceleration. It runs perfectly fine on a Linux Ubuntu Dekstop, but is quite slow on OS X Macbook Pro. On the other hand, Franka envs were fast on all machines we tested on.


### Data collection:

Example of collecting 1K episodes from FrankaTorque env:

```
python svae-dc/svae_dc/collect_env_experience.py \
 --env_name=FrankaTorque-v2 \
 --controller_class=WaypointsMinJerkPolicy \
 --output_prefix=experience/
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
 --log_interval=1 --save_interval=1 \
 --video_steps_interval=25 \
 --output_prefix=svaedata/
```
See ```svae-dc/svae_dc/svae_dc_args.py``` for more arguments.

All the SVAE-DC (NN) training is implemented in PyTorch. For visualization we use TensorboardX module to get a nice visual output for training. Start tensorboard:
```
tensorboard --logdir=svaedata/ --port=6006
```

The you will see some plots on the PLOTS tab. IMAGES tab would have reconstruction and generation videos. They should looks similar to this example image:

TODO: include tensorboard snapshot




### Bayesian Optimization:

TODO


<hr />
The above installation and demo instruction have been tested in OS X 10.14.5 (Mojave) and Linux Ubuntu 18.04.
