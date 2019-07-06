#
# PyBullet gym env for ABB Yumi robot.
#
import os
import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

from gym_bullet_extensions.bullet_manipulator import BulletManipulator
from gym_bullet_extensions.envs.manipulator_env import ManipulatorEnv


class YumiEnv(ManipulatorEnv):
    TABLE_MIN_X = 0.15; TABLE_MAX_X = 0.77
    TABLE_MIN_Y = -0.475; TABLE_MAX_Y = 0.475
    TABLE_WIDTH = TABLE_MAX_Y-TABLE_MIN_Y
    TABLE_HEIGHT = 0.08
    GRIPPER_LENGTH = 0.09
    WORKSPACE_MAX_Z = 0.50

    def __init__(self, num_objects, max_episode_steps,
                 control_mode='velocity', visualize=False, debug_level=0):
        self.debug_level = debug_level
        right_rest_arm_qpos =[0.5302, -1.8081, -1.3030, 0.3174,
                              -2.0786, 0.8680, 3.3587]
        left_rest_arm_qpos = [-0.4479, -1.2683, 1.9586, 0.4285,
                              0.9854, -0.0242, -0.5602]
        robot = BulletManipulator(
            os.path.join('yumi_robot', 'urdf', 'yumi_small_fingers.urdf'),
            control_mode=control_mode,
            ee_joint_name='yumi_joint_6_r', ee_link_name='yumi_link_7_r',
            base_pos=[0,0,-YumiEnv.TABLE_HEIGHT],
            rest_arm_qpos=right_rest_arm_qpos,
            left_ee_joint_name='yumi_joint_6_l',
            left_ee_link_name='yumi_link_7_l',
            left_fing_link_prefix='gripper_l', left_joint_suffix='_l',
            left_rest_arm_qpos=left_rest_arm_qpos,
            dt=1.0/500.0,
            kp=([100.0]*7 + [1.0]*2)*2,
            kd=([2.0]*7 + [0.1]*2)*2,
            min_z=0.09,
            visualize=visualize)
        assert(num_objects<=2)
        table_minmax_x_minmax_y = np.array([0.15, 0.77, -0.475, 0.475])
        super(YumiEnv, self).__init__(
            robot, num_objects, table_minmax_x_minmax_y,
            'cylinder_papertowel.urdf', max_episode_steps, visualize,
            debug_level)

    def get_all_init_object_poses(self):
        # 1st object starts at x=0.25 2nd ends at x=0.48
        all_init_object_poses = np.array([
            [0.31,-0.30,0.11], [0.43,-0.30,0.11]])
        return all_init_object_poses

    def get_init_pos(self):
        ee_pos = np.array([0.3, -0.5, 0.25])
        ee_quat = np.array(self.robot.sim.getQuaternionFromEuler(
            [np.pi,-np.pi/2,0]))
        fing_dist = self.robot.get_max_fing_dist()
        init_qpos = self.robot.ee_pos_to_qpos(
            ee_pos, ee_quat, fing_dist=fing_dist)
        assert(init_qpos is not None)
        # Use a manual qpos (to avoid instabilities of IK solutions).
        init_qpos[0:7] = np.array([0.83101, -1.500, -1.2019, -0.1817,
                                   -2.1447, -0.7756, 3.5720])
        return init_qpos, ee_pos, ee_quat, fing_dist
