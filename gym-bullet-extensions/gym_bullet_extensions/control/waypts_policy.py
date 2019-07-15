#
# Simple useful structured policies for manipulators.
# All parameter initialization is done in the constructor; get_action()
# returns the desired action based on observations, time or system state.
#
# For the vast majority of manipulators the kinematic model of the robot is
# known. So policies below generate desired targets in EE space and use Jacobian
# and/or IK to transform these points to desired joint angles or torques.
#
# The policies below assume that 'robot' passed to the constructor is an
# instance of BulletManipulator class. They use it to access the state of
# the robot and Jacobian info.
#
# @contactrika
#
import numpy as np
from scipy.signal import savgol_filter

import pybullet

from .control_util import (
    plan_min_jerk_trajectory, quaternion_to_euler, euler_to_quaternion,
    plan_linear_orientation_trajectory
)


class WaypointsVelPolicy:
    NUM_WAYPTS = 6
    WAYPT_DIM = 7+1  # 7DoF manipulator arm, timing param for vel
    DIM = NUM_WAYPTS*WAYPT_DIM

    def __init__(self, params, t_max, robot, get_init_pos_fxn):
        self.t_max = t_max
        assert(robot.control_mode == 'velocity')
        self.robot = robot
        self.lows, self.highs = WaypointsVelPolicy.vel_regions(
            robot, WaypointsVelPolicy.NUM_WAYPTS,
            WaypointsVelPolicy.WAYPT_DIM)
        params = params.reshape(WaypointsVelPolicy.NUM_WAYPTS,
                                WaypointsVelPolicy.WAYPT_DIM)
        unscaled_params = np.copy(params)*(self.highs-self.lows) + self.lows
        self.traj, self.vels, self.steps_per_wpt = WaypointsVelPolicy.make_traj(
            unscaled_params, t_max)

    @staticmethod
    def vel_regions(robot, num_waypts, dim):
        # The regions for this policy are conservative joint velocity limits
        # for moving (right) arm of a manipulator.
        num_r_joints = WaypointsVelPolicy.WAYPT_DIM-1
        ctrl_lows = np.zeros([num_waypts, dim])
        ctrl_highs = np.zeros([num_waypts, dim])
        min_vel = -1.0*robot.get_maxvel()[0:num_r_joints]
        max_vel = robot.get_maxvel()[0:num_r_joints]
        for wpt in range(num_waypts):
            ctrl_lows[wpt,:-1] = min_vel; ctrl_highs[wpt,:-1] = max_vel
        ctrl_lows[:,-1] = 0; ctrl_highs[:,-1] = 1  # timing param
        return ctrl_lows, ctrl_highs

    @staticmethod
    def make_traj(params, t_max):
        #print('make_traj: params', params)
        vels = np.copy(params[:,:-1])
        if np.sum(params[:,-1])<=0:
            timings = np.ones_like(params[:,-1])/params.shape[0]
        else:
            timings = params[:,-1]/np.sum(params[:,-1])  # timings to fracs
        steps_per_wpt = (np.ceil(t_max*timings)).astype(int)  # steps per waypt
        raw_traj = np.zeros([t_max, params.shape[1]-1])
        wpt = 0; wpt_st = 0
        for cum_st in range(t_max):
            raw_traj[cum_st] = params[wpt,:-1]
            wpt_st += 1
            if wpt_st > steps_per_wpt[wpt] and wpt+1<params.shape[0]:
                wpt += 1; wpt_st = 0  # next vel wpt
        # Smooth trajectory with a low-pass filter.
        winsz = int(steps_per_wpt.mean())
        if winsz%2==0: winsz += 1  # need odd window size for the filter
        polynomial_order = 3  # to model transitions from neg to pos vel
        #print('steps_per_wpt', steps_per_wpt)
        #print('applying savgol_filter winsz', winsz)
        traj = np.zeros_like(raw_traj)
        for jid in range(raw_traj.shape[1]):
            traj[:,jid] = savgol_filter(raw_traj[:,jid], winsz, polynomial_order)
        #print('traj', traj)
        return traj, vels, steps_per_wpt

    def get_action(self, obs, t=None):
        action = np.zeros([self.robot.info.dof])
        if t<self.t_max: action[0:7] = self.traj[t]  # vels for right arm
        for jid in self.robot.info.finger_jids_lst:  # finger closed for now
            action[jid] = -self.robot.info.joint_maxvel[jid]
        return action

    def print(self):
        print('WaypointsVelPolicy', self.steps_per_wpt, '\n', self.vels)


class WaypointsPosPolicy:
    NUM_WAYPTS = 6
    WAYPT_DIM = 9 # (3+3+1+2): EE pos, euler, fing_dist, kp, kd
    DIM = NUM_WAYPTS*WAYPT_DIM

    def __init__(self, params, t_max, robot, get_init_pos_fxn):
        self.t_max = t_max
        self.robot = robot
        is_yumi = 'yumi' in robot.info.joint_names[0]
        self.lows, self.highs = WaypointsPosPolicy.waypts_regions(
            robot, WaypointsPosPolicy.NUM_WAYPTS,
            WaypointsPosPolicy.WAYPT_DIM, yumi=is_yumi)
        params = params.reshape(WaypointsPosPolicy.NUM_WAYPTS,
                                WaypointsPosPolicy.WAYPT_DIM)
        unscaled_params = np.copy(params)*(self.highs-self.lows) + self.lows
        self.waypts = WaypointsPosPolicy.make_waypts(unscaled_params, is_yumi)
        traj, ee_pos_traj, ee_quat_traj = self.make_traj(
            self.waypts, t_max, robot, get_init_pos_fxn)
        self.traj = np.copy(traj)
        self.ee_pos_traj = np.copy(ee_pos_traj)
        self.ee_quat_traj = np.copy(ee_quat_traj)

    @staticmethod
    def waypts_regions(robot, num_waypts, dim, yumi=True):
        ctrl_lows = np.zeros([num_waypts, dim])
        ctrl_highs = np.zeros([num_waypts, dim])
        robot_pos, robot_quat = robot.sim.getBasePositionAndOrientation(
            robot.info.robot_id)
        # Region in front of the robot. Restricted to be relatively low but
        # above the ground (table) to avoid random motions outside of the main
        # workspace.
        if yumi:
            ctrl_lows[:,0] = 0.30; ctrl_highs[:,0] = 0.45   # x
            ctrl_lows[:,1] = -0.40; ctrl_highs[:,1] = 0.0  # y
            ctrl_lows[:,2] = 0.3; ctrl_highs[:,2] = 0.4   # z
            ctrl_lows[:,6] = 0; ctrl_highs[:,6] = robot.get_max_fing_dist()
            min_kp = 100.0; max_kp = 1000.0; min_kd = 10.0; max_kd = 100.0
        else:
            ctrl_lows[:,0] = 0.2; ctrl_highs[:,0] = 0.65   # x
            ctrl_lows[:,1] = -0.40; ctrl_highs[:,1] = 0.4  # y
            ctrl_lows[:,2] = 0.1; ctrl_highs[:,2] = 0.25   # z
            min_kp = 50.0; max_kp = 500.0; min_kd = 10.0; max_kd = 50.0
        robot_pos = np.array(robot_pos)
        ctrl_lows[:,0:3] += robot_pos; ctrl_highs[:,0:3] += robot_pos
        # gripper orientation encoded in euler angles
        ctrl_lows[:,3:6] = -np.pi;  ctrl_highs[:,3:6] = np.pi
        ctrl_lows[:,6] = 0; ctrl_highs[:,6] = robot.get_max_fing_dist()
        ctrl_lows[:,7] = min_kp; ctrl_highs[:,7] = max_kp
        ctrl_lows[:,8] = min_kd; ctrl_highs[:,8] = max_kd
        return ctrl_lows, ctrl_highs

    @staticmethod
    def make_waypts(prms, yumi=True):
        # Initialize EE waypoints appropriate to search for a pushing strategy.
        # Restrict the gripper to point mostly down (and forward for yumi).
        rpys = prms[:,3:6]
        small_angle = np.pi/16
        if yumi:  # restrict pitch (forward)
            fwd_rpy = np.array([np.pi,-np.pi/2,0])
            rpys = np.clip(rpys, fwd_rpy-small_angle, fwd_rpy+small_angle)
        else:
            rid = 0  #  restrict roll (down)
            almost_down = np.pi-small_angle
            nondown_ids = np.where(np.abs(rpys[:,rid])<almost_down)
            rpys[nondown_ids,rid] = almost_down*np.sign(rpys[nondown_ids,rid])
            max_rpy = np.array([np.pi,small_angle,small_angle])
            rpys = np.clip(rpys, -max_rpy, max_rpy)
        prms[:,3:6] = rpys
        # Fingers closed for now.
        prms[:,6] = 0.0
        return prms

    def make_traj(self, waypts, t_max, robot, get_init_pos_fxn):
        num_waypts = waypts.shape[0]
        # Convert ee waypoints to space of joint angles.
        steps_per_waypt = int(t_max/(num_waypts))
        prev_qpos, prev_ee_pos, prev_ee_quat, _ = get_init_pos_fxn()
        prev_gains = None
        assert(prev_qpos is not None)
        traj = np.zeros([t_max, prev_qpos.shape[0]+2])
        ee_pos_traj = np.zeros([t_max, 3])
        ee_quat_traj = np.zeros([t_max, 4])
        t = 0
        print('make_traj for policy'); self.print()
        for wpt in range(num_waypts):
            ee_pos, ee_quat, fing_dist, gains = self.parse_waypoint(waypts[wpt])
            qpos = robot.ee_pos_to_qpos(ee_pos, ee_quat, fing_dist)
            if qpos is None:  # move up waypoint to not collide with the table
                adjusted_ee_pos = np.copy(ee_pos); max_z = self.highs[wpt,2]
                while (adjusted_ee_pos[2]<max_z) and (qpos is None):
                    adjusted_ee_pos[2] += 0.005  # move up 0.5cm
                    qpos = robot.ee_pos_to_qpos(
                        adjusted_ee_pos, ee_quat, fing_dist)
                    if qpos is not None:
                        print('ee_pos', ee_pos, 'adjusted_ee_pos', adjusted_ee_pos)
                        ee_pos = adjusted_ee_pos; break
            if qpos is None:  # if all else fails: just use previous waypt
                print('using prev waypt instead of', ee_pos, ee_quat)
                qpos = prev_qpos; ee_pos = prev_ee_pos; ee_quat = prev_ee_quat
            assert(qpos is not None)
            ee_pos_traj[t:t+steps_per_waypt,:] = ee_pos[:]
            ee_quat_traj[t:t+steps_per_waypt,:] = ee_quat[:]
            traj[t:t+steps_per_waypt,0:-2] = qpos[:]
            traj[t:t+steps_per_waypt,-2:] = gains[:]
            t += steps_per_waypt
            prev_qpos = qpos; prev_ee_pos = ee_pos; prev_ee_quat = ee_quat
            prev_gains = gains
        if t < t_max:  # set remaining to last entry
            traj[t:,0:-2] = prev_qpos[:]
            traj[t:,-2:] = prev_gains[:]
            ee_pos_traj[t:,:] = prev_ee_pos[:]
            ee_quat_traj[t:,:] = prev_ee_quat[:]
        return traj, ee_pos_traj, ee_quat_traj

    def get_action(self, obs, t=None):
        if t is None: assert(False)  # specify t for time-varying policy
        assert(t>=0)  # but t>self.t_max ok, will just return last ctrl
        if t >= self.t_max: t = self.t_max-1
        des_qpos = self.traj[t,0:-2]; kp = self.traj[t,-2]; kd = self.traj[t,-1]
        if self.robot.control_mode == 'ee_position':
            ee_pos = self.ee_pos_traj[t]
            ee_quat = self.ee_quat_traj[t]
            fing_dist = 0.0  # fingers closed for now
            action = np.hstack([ee_pos, ee_quat, fing_dist])
            return action
        elif self.robot.control_mode == 'position':
            return des_qpos
        elif self.robot.control_mode == 'torque':
            qpos = self.robot.get_qpos(); qvel = self.robot.get_qvel()
            torques = kp*(des_qpos - qpos) + kd*(0 - qvel)
            return torques
        else:
            print('Unknown control_mode', self.robot.control_mode)
            assert(False)  # unknown control mode

    def parse_waypoint(self, waypoint):  # EE pos, quat, fing_dist, gains(kp,kd)
        ee_quat = np.array(pybullet.getQuaternionFromEuler(waypoint[3:6]))
        return waypoint[0:3], ee_quat, waypoint[6], waypoint[7:9]

    def print(self):
        print('WaypointsPosPolicy\n', self.waypts)


class WaypointsEEPolicy(WaypointsPosPolicy):
    NUM_WAYPTS = 6
    WAYPT_DIM = 7 # (3+3+1): EE pos, euler, fing_dist, kp, kd
    DIM = NUM_WAYPTS*WAYPT_DIM

    def __init__(self, params, t_max, robot, get_init_pos_fxn):
        # Not using gains, so put in zeros, but otherwise the same superclass
        params = params.reshape(WaypointsEEPolicy.NUM_WAYPTS,
                                WaypointsEEPolicy.WAYPT_DIM)
        params_wzero_gains = np.hstack(
            [params, np.zeros([WaypointsEEPolicy.NUM_WAYPTS,2])])
        super(WaypointsEEPolicy, self).__init__(
            params_wzero_gains, t_max, robot, get_init_pos_fxn)

    def print(self):
        print('WaypointsEEPolicy\n', self.waypts)


class WaypointsMinJerkPolicy(WaypointsPosPolicy):
    def __init__(self, params, t_max, robot, get_init_pos_fxn):
        super(WaypointsMinJerkPolicy, self).__init__(
            params, t_max, robot, get_init_pos_fxn)

    def make_traj(self, waypts, t_max, robot, get_init_pos_fxn):
        assert(robot.control_mode == 'torque')  # need torque-controlled robot
        # Compute minimum jerk trajectory
        _, prev_ee_pos, prev_ee_quat, _, = get_init_pos_fxn()
        prev_fing_dist = robot.get_fing_dist()
        # Note: using custom function to convert to euler in order to match
        # pybullet's conventions on how angular velocity is reported.
        prev_ee_orient = quaternion_to_euler(prev_ee_quat)
        num_waypts = waypts.shape[0]
        steps_per_waypt = int(t_max/(num_waypts))
        traj = np.zeros([t_max, 3*4+2])
        ee_quat_traj = np.zeros([num_waypts*steps_per_waypt, 4])
        t = 0
        for wpt in range(num_waypts):
            ee_pos, ee_quat, fing_dist, gains = self.parse_waypoint(waypts[wpt])
            ee_orient = quaternion_to_euler(ee_quat)
            qpos = robot.ee_pos_to_qpos(ee_pos, ee_quat, fing_dist)
            if qpos is None: # internal checks (e.g. collisions); qpos not used
                ee_pos = prev_ee_pos; ee_orient = prev_ee_orient
                fing_dist = prev_fing_dist
            Y, Yd, Ydd = plan_min_jerk_trajectory(
                prev_ee_pos, ee_pos, steps_per_waypt*robot.dt, robot.dt)
            th = plan_linear_orientation_trajectory(
                prev_ee_orient, ee_orient, steps_per_waypt)
            traj[t:t+steps_per_waypt,0:3] = Y[:]
            traj[t:t+steps_per_waypt,3:6] = Yd[:]
            traj[t:t+steps_per_waypt,6:9] = Ydd[:]
            traj[t:t+steps_per_waypt,9:12] = th[:]
            traj[t:t+steps_per_waypt,12:] = gains[:]
            for substep in range(steps_per_waypt):
                ee_quat_traj[t+substep,:] = euler_to_quaternion(th[substep])
            t += steps_per_waypt
            prev_ee_pos = ee_pos; prev_ee_orient = ee_orient
            prev_fing_dist = fing_dist
        if t+1<self.t_max: traj[t+1:,:] = traj[t,:]  # set rest to last entry
        return traj, traj[:,0:3], ee_quat_traj

    def get_action(self, obs, t=None, left=False):
        if t is None: assert(False)  # specify t for time-varying policy
        assert(t>=0)  # but t>self.t_max ok, will just return last ctrl
        if t >= self.t_max: t = self.t_max-1
        action = self.traj[t]
        des_ee_pos = action[0:3]
        des_ee_vel = action[3:6]
        des_ee_acc = action[6:9]
        des_ee_orient = action[9:12]
        kp = action[12]; kd = action[13]
        curr_ee_pos, curr_ee_quat, curr_ee_linvel, curr_ee_angvel = \
            self.robot.get_ee_pos_ori_vel()
        des_ee_force = (kp*(des_ee_pos - curr_ee_pos) +
                        kd*(des_ee_vel - curr_ee_linvel) + des_ee_acc)
        curr_ee_orient = quaternion_to_euler(curr_ee_quat)
        des_ee_torque = (1.0*(des_ee_orient - curr_ee_orient) -
                         0.1*curr_ee_angvel)
        J_lin, J_ang = self.robot.get_ee_jacobian(left=left)
        jacobian = np.vstack([J_lin, J_ang])
        torque = np.matmul(jacobian.transpose(),
                           np.hstack([des_ee_force, des_ee_torque]))
        # Fingers closed for this policy.
        fing_dist = self.robot.get_fing_dist()
        maxforce = self.robot.info.joint_maxforce
        for jid in self.robot.info.finger_jids_lst:
            torque[jid] = 0 if fing_dist<=0.001 else -maxforce[jid]/2
        return torque

    def print(self):
        print('WaypointsMinJerkTorquePolicy\n', self.waypts)
