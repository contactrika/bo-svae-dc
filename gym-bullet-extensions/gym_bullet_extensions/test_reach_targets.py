#
# A quick script to test pybullet envs.
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


def test_robot(robot):
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


def test_franka():
    robot_file = os.path.join('franka_robot', 'fullfranka.urdf')
    robot = BulletManipulator(
        robot_file, ee_joint_name='panda_joint7', ee_link_name='panda_hand',
        visualize=True)
    test_robot(robot)


def main():
    test_franka()


if __name__ == "__main__":
    main()