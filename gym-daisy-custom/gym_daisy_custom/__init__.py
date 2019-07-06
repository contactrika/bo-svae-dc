import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

for variant in ['', 'Long']:
    max_episode_steps = 3000 if variant=='Long' else 1000
    for penalize_stalled in [0, 1]:
        for debug_level in [0, 1, 2]:
            suffix = 'Hw' if penalize_stalled==0 else ''
            if debug_level>0 and debug_level<=1:
                suffix += 'Debug'
            elif debug_level>=2:
                suffix += 'Viz'
            id_str = 'DaisyCustom'+variant+suffix+'-v0'
            register(
                id=id_str,
                entry_point='gym_daisy_custom.envs:DaisyCustomEnv',
                max_episode_steps=max_episode_steps,
                nondeterministic=True,
                kwargs={'max_episode_steps':max_episode_steps,
                        'can_terminate_early':(variant=='Long'),
                        'penalize_stalled':penalize_stalled,
                        'visualize':(debug_level>=2), 'debug_level':debug_level},
            )
