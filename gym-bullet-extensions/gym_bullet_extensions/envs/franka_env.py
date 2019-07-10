#
# PyBullet gym env for Franka Emika robot.
#
# @contactrika
#
import os

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

from gym_bullet_extensions.bullet_manipulator import BulletManipulator
from gym_bullet_extensions.envs.manipulator_env import ManipulatorEnv


class FrankaEnv(ManipulatorEnv):
    def __init__(self, num_objects, max_episode_steps,
                 control_mode='velocity', visualize=False, debug_level=0):
        self.pos_only_state = True  # don't include velocities in state
        robot = BulletManipulator(
            os.path.join('franka_robot', 'fullfranka.urdf'),
            control_mode=control_mode,
            ee_joint_name='panda_joint7', ee_link_name='panda_hand',
            base_pos=[0,0,0], dt=1.0/500.0,
            kp=([200.0]*7 + [1.0]*2),
            kd=([2.0]*7  + [0.1]*2),
            min_z=0.00,
            visualize=visualize)
        table_minmax_x_minmax_y = np.array([0.0, 1.50, -0.45, 0.45])
        super(FrankaEnv, self).__init__(
            robot, num_objects, table_minmax_x_minmax_y, 'cylinder_block.urdf',
            max_episode_steps, visualize, debug_level)
        self.set_join_limits_for_forward_workspace()

    def set_join_limits_for_forward_workspace(self):
        # Set reasonable joint limits for operating the space mainly in front
        # of the robot. Our main workspace is the table in front of the robot,
        # so we are not interested in exploratory motions outside of the main
        # workspace.
        minpos = np.copy(self.robot.get_minpos())
        maxpos = np.copy(self.robot.get_maxpos())
        # operate in the workspace in front of the robot
        minpos[0] = -0.5; maxpos[0] = 0.5
        minpos[1] = 0.0
        minpos[2] = -0.5; maxpos[2] = 0.5
        #minpos[3] = -3.0; maxpos[3] = -1.0  # don't stretch out the elbo
        self.robot.set_joint_limits(minpos, maxpos)

    def get_init_pos(self):
        ee_pos = np.array([0.25,0.30,0.30])
        ee_quat = np.array(self.robot.sim.getQuaternionFromEuler([np.pi,0,0]))
        fing_dist = 0.0
        #init_qpos = self.robot.ee_pos_to_qpos(
        #    ee_pos, ee_quat, fing_dist=fing_dist)
        #assert(init_qpos is not None)
        # Use a manual qpos (to avoid instabilities of IK solutions).
        init_qpos = np.array([ 0.4239,  0.,      0.4799, -2.727,
                               0.2047,  2.4689,  1.5125,  0., 0.])
        return init_qpos, ee_pos, ee_quat, fing_dist

    def get_all_init_object_poses(self):
        all_init_object_poses = np.array([
            [0.32,0.15,0.11], [0.50,0.15,0.11]])
        return all_init_object_poses

    def get_is_bad(self, debug=False):
        bad = super(FrankaEnv, self).get_is_bad(debug)
        return bad
