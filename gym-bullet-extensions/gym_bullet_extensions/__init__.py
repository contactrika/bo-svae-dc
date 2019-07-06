import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

ctrl_modes = {'EEPos':'ee_position',
              'Pos':'position', 'Vel':'velocity', 'Torque':'torque'}
max_num_obj = 2
for env_base in ['Franka', 'Yumi']:
    max_episode_steps = 500 if env_base=='Franka' else 1000
    for ctrl_nm, ctrl_mode in ctrl_modes.items():
        nm_str = env_base+ctrl_nm
        for num_obj in range(max_num_obj+1):
            for debug_level in [0, 1, 2]:
                suffix = ''
                if debug_level>0:
                    suffix = 'Debug' if debug_level<=1 else 'Viz'
                register(
                    id=nm_str+suffix+'-v'+str(num_obj),
                    entry_point='gym_bullet_extensions.envs:'+env_base+'Env',
                    max_episode_steps=max_episode_steps,
                    reward_threshold=1.0,
                    nondeterministic=True,
                    kwargs={'num_objects':num_obj,
                            'max_episode_steps':max_episode_steps,
                            'control_mode':ctrl_mode,
                            'visualize':(debug_level>=2),
                            'debug_level':debug_level})
