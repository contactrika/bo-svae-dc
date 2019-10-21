#
# A quick script to test pybullet envs.
#
# @contactrika
#
import os

import numpy as np
import pybullet

from bullet_manipulator import BulletManipulator


def reach_eepos(robot, tgt_ee_pos, tgt_ee_quat, fing_dist,
                tgt_viz_id, num_steps=100):
    tgt_ee_pos = np.array(tgt_ee_pos); tgt_ee_quat = np.array(tgt_ee_quat)
    print('tgt_ee_pos', tgt_ee_pos, 'fing_dist', fing_dist)
    robot.sim.resetBasePositionAndOrientation(tgt_viz_id, tgt_ee_pos, tgt_ee_quat)
    qpos = robot._ee_pos_to_qpos_raw(tgt_ee_pos, tgt_ee_quat, fing_dist)
    print('target qpos', qpos)
    qpos_ok = robot.ee_pos_to_qpos(tgt_ee_pos, tgt_ee_quat, fing_dist)
    print('qpos_ok', qpos_ok)
    robot.refresh_viz()
    input('----------------------- Press Enter to reset_to_qpos()')
    robot.reset_to_qpos(qpos)
    robot.refresh_viz()
    input('-------------------------------------------- Press Enter to reset()')
    robot.reset()
    robot.refresh_viz()
    input('---------------- Press Enter to do pybullet PD control')
    tgt_qpos = qpos
    tgt_qvel = np.zeros_like(tgt_qpos)
    for i in range(num_steps):
        robot.move_to_qposvel(
            tgt_qpos, tgt_qvel, mode=pybullet.PD_CONTROL, kp=100.0, kd=10.0)
    robot.refresh_viz()
    print('Final qpos', robot.get_qpos())
    ee_pos, ee_quat, ee_vel, _ = robot.get_ee_pos_ori_vel()
    ee_euler = robot.sim.getEulerFromQuaternion(ee_quat)
    tgt_ee_euler = robot.sim.getEulerFromQuaternion(tgt_ee_quat)
    print('Final ee pos quat vel', ee_pos, ee_quat, ee_vel)
    print('tgt_ee_euler', tgt_ee_euler, '\nvs final    ', ee_euler)
    input('---------------- Done. Press Enter to go on')
    robot.reset()
    robot.refresh_viz()


def test_franka():
    robot_file = os.path.join('franka_robot', 'franka_small_fingers.urdf')
    robot = BulletManipulator(
        robot_file, control_mode='position',
        ee_joint_name='panda_joint7', ee_link_name='panda_hand',
        base_pos=[0,0,0], dt=1.0/500.0,
        kp=([200.0]*7 + [1.0]*2), kd=([2.0]*7 + [0.1]*2),
        min_z=0.00, visualize=True)
    tgt_viz_id = robot.create_visual_area(pybullet.GEOM_CYLINDER, [0,0,0],
                                          radius=0.03, rgba=[0, 1, 1, 0.5])
    max_fing_dist = 0.08
    reach_eepos(robot, [0.4,0.0,0.4], [1,0,0,0], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.4,0.1,0.3], [1,0,0,0], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.3,0.2,0.3], [1,0,0,0], 0.02, tgt_viz_id)
    reach_eepos(robot, [0.4,0.0,0.4], [0,0,0,1], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.1,-0.3,0.7], [0,1,0,0], 0.04, tgt_viz_id)
    reach_eepos(robot, [0.1,0.3,0.7], [0,0,1,0], 0.08, tgt_viz_id)
    print('------------------ Random targets --------------------')
    for i in range(10):
        reach_eepos(robot, np.random.rand(3)-[0.5,0.5,0], [1,0,0,0],
                    max_fing_dist*(float(i+1)/10), tgt_viz_id)
    print('------------------ Collision targets --------------------')
    reach_eepos(robot, [0.4,0.0,0.05], [1,0,0,0], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.4,0.1,0.05], [1,0,0,0], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.3,0.2,0.05], [1,0,0,0], 0.02, tgt_viz_id)


def test_yumi():
    robot_file = os.path.join('yumi_robot', 'urdf', 'yumi_small_fingers.urdf')
    right_rest_arm_qpos =[0.5302, -1.8081, -1.3030, 0.3174,
                          -2.0786, 0.8680, 3.3587]
    left_rest_arm_qpos = [-0.4479, -1.2683, 1.9586, 0.4285,
                          0.9854, -0.0242, -0.5602]
    robot = BulletManipulator(
        robot_file, control_mode='position',
        ee_joint_name='yumi_joint_6_r', ee_link_name='yumi_link_7_r',
        base_pos=[0,0,-0.008], rest_arm_qpos=right_rest_arm_qpos,
        left_ee_joint_name='yumi_joint_6_l',
        left_ee_link_name='yumi_link_7_l',
        left_fing_link_prefix='gripper_l', left_joint_suffix='_l',
        left_rest_arm_qpos=left_rest_arm_qpos,
        kp=([100.0]*7 + [1.0]*2)*2, kd=([2.0]*7 + [0.1]*2)*2,
        dt=1.0/500.0, min_z=0.09, visualize=True)
    tgt_viz_id = robot.create_visual_area(pybullet.GEOM_CYLINDER, [0,0,0],
                                          radius=0.03, rgba=[0, 1, 1, 0.5])
    max_fing_dist = 0.04
    ee_quat = np.array(robot.sim.getQuaternionFromEuler([np.pi,-np.pi/2,0]))
    reach_eepos(robot, [0.4,0.0,0.3], ee_quat, 0.00, tgt_viz_id)
    reach_eepos(robot, [0.4,-0.1,0.2], ee_quat, 0.00, tgt_viz_id)
    reach_eepos(robot, [0.3,-0.2,0.2], ee_quat, 0.02, tgt_viz_id)
    reach_eepos(robot, [0.2,0.0,0.3], ee_quat, 0.00, tgt_viz_id)
    reach_eepos(robot, [0.2,-0.3,0.4], ee_quat, 0.04, tgt_viz_id)
    reach_eepos(robot, [0.2,0.3,0.4], ee_quat, 0.08, tgt_viz_id)
    print('------------------ Random targets --------------------')
    for i in range(10):
        reach_eepos(robot, np.random.rand(3)-[0.5,0.5,0], ee_quat,
                    max_fing_dist*(float(i+1)/10), tgt_viz_id)
    print('------------------ Collision targets --------------------')
    reach_eepos(robot, [0.4,0.0,0.05], [1,0,0,0], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.4,-0.1,0.05], [1,0,0,0], 0.00, tgt_viz_id)
    reach_eepos(robot, [0.3,-0.2,0.05], [1,0,0,0], 0.02, tgt_viz_id)


def main():
    test_franka()
    test_yumi()


if __name__ == "__main__":
    main()