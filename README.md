## SVAE-DC code for CoRL2019 submission (anonymized)

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
```

If pybullet simulator fails to install on your machine, sometimes it is easier to install it from source. For installation troubleshooting and further information on PyBullet see https://pybullet.org

Our code for the Daisy environment we used in our experiments is included in ```gym-daisy-custom``` module. However, the base class it uses for the pybullet Daisy simulator is in the process of being open-sourced by another group (should be out shortly). So all our work is included in this repo, but at the moment there is no way to give the public access to the lower-level pybullet Daisy simulation code.

Install SVAE-DC training and BO repo:

```
cd svae-dc
pip install -e .
```

### Visualizing Yumi and Franka envs:

TODO

### Data collection:

TODO

### SVAE-DC training:

TODO

TODO: include tensorboard snapshot

### Bayesian Optimization:

TODO

