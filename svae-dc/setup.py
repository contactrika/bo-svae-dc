from setuptools import setup

setup(name='svae_dc',
      version='0.1',
      description='Sequential VAE with Dynamic Compression',
      packages=['svae_dc'],
      install_requires=['gym', 'torch', 'numpy', 'moviepy', 'gpytorch', 'botorch==0.1.2', 'tensorboardX'],
      )
