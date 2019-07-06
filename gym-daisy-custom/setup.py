from setuptools import setup

setup(name='daisy_custom_env',
      version='0.1',
      description='Custom PyBullet env for Daisy robot.',
      packages=['gym_daisy_custom'],
      install_requires=['gym', 'pybullet'])
