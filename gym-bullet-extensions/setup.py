from setuptools import setup

setup(name='gym_bullet_extensions',
      version='0.1',
      description='PyBullet envs with ABB Yumi and Franka Emika.',
      packages=['gym_bullet_extensions'],
      install_requires=['numpy', 'gym', 'pybullet'])
