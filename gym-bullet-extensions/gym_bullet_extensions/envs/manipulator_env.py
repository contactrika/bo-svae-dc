#
# PyBullet gym env for manipulators and movable objects.
#
# @contactrika
#
from copy import copy

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import gym
from gym import spaces
import pybullet

from gym_bullet_extensions.bullet_manipulator import BulletManipulator


class ManipulatorEnv(gym.Env):
    MAX_NUM_OBJECTS = 4
    OBJECT_ORI_THRESH = 0.5  # how far away from vertical is ok (~30deg)
    OBJECT_MIN = np.array([-1,-1,0]+[-1]*3+[-1]*3)  # pos, euler sin, euler cos
    OBJECT_MAX = np.array([2,2,1]+[1]*3+[1]*3)      # pos, euler sin, euler cos
    OBJECT_MASS = 0.150
    OBJECT_RESTITUTION = 0.05
    OBJECT_LATERAL_FRICTION = 0.3
    OBJECT_ROLING_FRICTION = 0.01
    # Randomize object properties. Pybullet simulations are extremely sensitive
    # to friction parameters, hence pick a sane scaling range. Note that on
    # some OS versions pybullet is non-deterministic even with the same
    # parameters (perhaps due to small numerical instabilities).
    OBJECT_MIN_SCALING = 0.95
    OBJECT_MAX_SCALING = 1.05

    def __init__(self, robot, num_objects, table_minmax_x_minmax_y, object_urdf,
                 max_episode_steps=300, visualize=False, debug_level=0):
        self.pos_only_state = True  # don't include velocities in state
        super(ManipulatorEnv, self).__init__()
        self.robot = robot
        num_joints = self.robot.get_minpos().shape[0]
        self.reset_qpos, _, _, _ = self.get_init_pos()
        self.debug_level = debug_level
        self.max_episode_steps = max_episode_steps
        self.visualize = visualize
        # Init object properties.
        self.object_init_poses, self.object_init_quats = \
            self.get_all_init_object_poses(num_objects)
        self.object_ids = self.robot.load_objects_from_file(
            object_urdf, self.object_init_poses, self.object_init_quats)
        self.object_mass = ManipulatorEnv.OBJECT_MASS
        self.object_restitution = ManipulatorEnv.OBJECT_RESTITUTION
        self.object_lateral_friction = ManipulatorEnv.OBJECT_LATERAL_FRICTION
        self.object_rolling_friction = ManipulatorEnv.OBJECT_ROLING_FRICTION
        # Report observation and action spaces.
        if self.pos_only_state:
            self.observation_space = spaces.Box(
                np.hstack([self.robot.get_minpos(),
                           np.tile(ManipulatorEnv.OBJECT_MIN,num_objects)]),
                np.hstack([self.robot.get_maxpos(),
                           np.tile(ManipulatorEnv.OBJECT_MAX,num_objects)]))
        else:
            self.observation_space = spaces.Box(
                np.hstack([self.robot.get_minpos(),
                           -BulletManipulator.MAX_VELOCITY*np.ones(num_joints),
                           np.tile(ManipulatorEnv.OBJECT_MIN,num_objects)]),
                np.hstack([self.robot.get_maxpos(),
                           BulletManipulator.MAX_VELOCITY*np.ones(num_joints),
                           np.tile(ManipulatorEnv.OBJECT_MAX,num_objects)]))
        act_low, act_high = self.robot.action_low_high_ranges()
        self.action_space = spaces.Box(act_low, act_high)
        print('observation_space', self.observation_space.low,
              self.observation_space.high)
        print('action_space', self.action_space.low, self.action_space.high)
        # Create a table area in front of the robot.
        self.table_min_x = table_minmax_x_minmax_y[0]
        self.table_max_x = table_minmax_x_minmax_y[1]
        self.table_min_y = table_minmax_x_minmax_y[2]
        self.table_max_y = table_minmax_x_minmax_y[3]
        table_half_x = (self.table_max_x-self.table_min_x)/2
        table_half_y = (self.table_max_y-self.table_min_y)/2
        table_center = [self.table_min_x+table_half_x, 0, -0.049]
        table_half_extents = [table_half_x, table_half_y, 0.05]
        self.robot.create_visual_area(pybullet.GEOM_BOX, table_center,
                                      table_half_extents, rgba=[0,0.1,0,1.0])
        # Create a list of names for obs for easier debugging.
        joint_names = self.robot.info.joint_names
        obs_names = [nm+'_pos' for nm in joint_names]
        if not self.pos_only_state:
            obs_names.extend([nm+'_vel' for nm in joint_names])
        for j in range(num_objects):
            for i in range(ManipulatorEnv.OBJECT_MIN.shape[0]):
                obs_names.extend(['obj'+str(j)+'_dim'+str(i)])
        self.obs_names = ['jnt_'+nm for nm in obs_names]

    def get_obs_names(self):
        return self.obs_names

    def reset(self):
        if self.visualize: input('Pres Enter to continue reset')
        self.stepnum = 0
        self.badlen = 0
        self.collided = False
        self.robot.reset_to_qpos(self.reset_qpos)
        self.robot.reset_objects(
            self.object_ids, self.object_init_poses, self.object_init_quats)
        return self.get_obs()

    def randomize(self, min_scaling=None, max_scaling=None, debug=False):
        # Slightly change mass, friction, restitution of the object.
        # This simulates non-determinism: uncertainty over object properties
        # in the real world. This yields non-deterministic simulation, hence
        # stochastic trajectories even for deterministic controllers.
        if min_scaling is None: min_scaling = ManipulatorEnv.OBJECT_MIN_SCALING
        if max_scaling is None: max_scaling = ManipulatorEnv.OBJECT_MAX_SCALING
        scalings = np.random.rand(len(self.object_ids), 4)
        scalings = scalings*(max_scaling-min_scaling)+min_scaling
        rnd_params = np.zeros([len(self.object_ids), 4])
        for objid in range(len(self.object_ids)):
            rnd_params[objid, 0] = ManipulatorEnv.OBJECT_MASS*scalings[objid, 0]
            rnd_params[objid, 1] = (ManipulatorEnv.OBJECT_RESTITUTION*
                                    scalings[objid, 1])
            rnd_params[objid, 2] = (ManipulatorEnv.OBJECT_LATERAL_FRICTION*
                                    scalings[objid, 2])
            rnd_params[objid, 3] = (ManipulatorEnv.OBJECT_ROLING_FRICTION*
                                    scalings[objid, 3])
        rnd_params = rnd_params.reshape(-1)  # flatten
        self.set_randomize(rnd_params, debug)
        return rnd_params

    def set_randomize(self, rnd_params, debug=False):
        assert(rnd_params.shape[0]==len(self.object_ids)*4)
        rnd_params = rnd_params.reshape(len(self.object_ids), -1)
        for objid in range(len(self.object_ids)):
            mass = rnd_params[objid, 0]
            restitution = rnd_params[objid, 1]
            lat_fric = rnd_params[objid, 2]
            rol_fric = rnd_params[objid, 3]
            # TODO: Does bullet change inertial matrix based on mass, shape
            # TODO: and type, or do we need to re-specify manually?
            self.robot.sim.changeDynamics(
                self.object_ids[objid], -1, mass=mass,
                lateralFriction=lat_fric,
                rollingFriction=rol_fric, restitution=restitution)
            if debug:
                msg = 'changeDynamics for obj {:d}: mass {:.4f}'
                msg += ' restitution {:.4f} lat_fric {:.4f} roll_fric {:.4f}'
                print(msg.format(objid, mass, restitution, lat_fric, rol_fric))

    def step(self, action):  # assume unscaled action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.apply_action(action)
        #if self.visualize: time.sleep(0.05)
        next_obs = self.get_obs()
        # Update internal counters.
        self.stepnum += 1
        is_bad = self.get_is_bad()
        if is_bad: self.badlen += 1
        # Report reward stats and other info.
        reward = 0.0
        done = (self.stepnum == self.max_episode_steps)
        info = {'is_bad':is_bad}
        if done:
            final_rwd = self.final_rwd()
            info['done_obs'] = copy(next_obs) # vector envs lose last frame
            info['done_reward'] = final_rwd
            info['done_badfrac'] = self.badlen/self.max_episode_steps
            if self.debug_level>=1:
                print('rwd {:0.4f} badlen {:d}'.format(final_rwd, self.badlen))
        return next_obs, reward, done, info

    def render(self, mode="rgb_array", close=False):
        pass  # implemented in pybullet

    def render_debug(self, width=600):
        return self.robot.render_debug(width=width)

    def get_obs(self):
        obs = []
        obs.append(self.robot.get_qpos())
        if not self.pos_only_state: obs.append(self.robot.get_qvel())
        for objid, obj in enumerate(self.object_ids):
            pos, ori = self.robot.sim.getBasePositionAndOrientation(obj)
            euler = self.robot.sim.getEulerFromQuaternion(ori)
            obs.append(np.clip(np.array(pos),
                               ManipulatorEnv.OBJECT_MIN[0:3],
                               ManipulatorEnv.OBJECT_MAX[0:3]))
            obs.append(np.sin(euler)); obs.append(np.cos(euler))
        obs = np.hstack(obs)
        eps = 1e-6
        if (obs<self.observation_space.low-eps).any():
            print('   obs', obs, '\nvs low', self.observation_space.low)
            assert(False)
        if (obs>self.observation_space.high+eps).any():
            print('    obs', obs, '\nvs high', self.observation_space.high)
            print(obs[np.where(obs>self.observation_space.high)])
            print(self.observation_space.high[np.where(
                obs>self.observation_space.high)])
            assert(False)
        return obs

    def override_state(self, obs):
        num_joints = self.robot.info.dof
        self.robot.reset_to_qpos(obs[:num_joints])
        ofst = num_joints
        if not self.pos_only_state: ofst += num_joints  # skip vels
        for objid, obj in enumerate(self.object_ids):
            obj_pos = obs[ofst:ofst+3]; ofst+=3
            obj_sin = obs[ofst:ofst+3]; ofst+=3
            obj_cos = obs[ofst:ofst+3]; ofst+=3
            obj_quat = pybullet.getQuaternionFromEuler(
                np.arctan2(obj_sin, obj_cos))
            self.robot.sim.resetBasePositionAndOrientation(
                obj, obj_pos, obj_quat)

    def final_rwd(self):
        # The goal of this env is to clear the workspace by moving objects
        # to the left (high y, so high obj_pos[1]) without tipping them over or
        # colliding with the table.
        y_lst = []
        table_width = self.table_max_y-self.table_min_y
        tip_rwd = -1.0*table_width  # penalize if tipped or fell off the table
        for objid, obj in enumerate(self.object_ids):
            obj_pos, obj_quat = pybullet.getBasePositionAndOrientation(obj)
            if self.debug_level>0: print('obj_pos', obj_pos)
            if not self.object_tipped(obj_quat) and not self.off_table(obj_pos):
                diff = obj_pos[1]-self.object_init_poses[objid,1]
                if self.object_init_poses[objid,1]>0: diff *= -1 # left-to-right
                y_lst.append(diff)
        if self.debug_level>0: print('y_lst', y_lst)
        num_tipped = len(self.object_ids)-len(y_lst)
        rwd = tip_rwd*num_tipped
        if len(y_lst)>0:
            rwd += np.array(y_lst).sum() # reward moving objects
        return rwd

    def get_is_bad(self, ee_eps=0.05, debug=False):
        ee_pos = self.robot.get_ee_pos()
        min_x = self.table_min_x-ee_eps; max_x = self.table_max_x+ee_eps
        min_y = self.table_min_y-ee_eps; max_y = self.table_max_y+ee_eps
        if (ee_pos[0]<min_x or ee_pos[0]>max_x or
            ee_pos[1]<min_y or ee_pos[1]>max_y):
            if debug: print('robot arm outside of workspace area')
            return True
        if ee_pos[2]<=self.robot.min_z:
            if debug: print('too close to the table: ee_pos', ee_pos)
            return True  # too close to the table
        qpos = self.robot.get_qpos()
        if not self.robot.collisions_ok(qpos):
            if debug: print('collision with table')
            return True  # collision
        for objid, obj in enumerate(self.object_ids):
            pos, quat = self.robot.sim.getBasePositionAndOrientation(obj)
            if self.off_table(pos):
                if debug: print('obj', objid, 'off table')
                return True
            if self.object_tipped(quat):
                if debug: print('obj', objid, 'tipped')
                return True
        return False  # everything is good

    def object_tipped(self, obj_quat):
        euler = np.array(self.robot.sim.getEulerFromQuaternion(obj_quat))
        if (np.abs(euler[0])>ManipulatorEnv.OBJECT_ORI_THRESH or
            np.abs(euler[1])>ManipulatorEnv.OBJECT_ORI_THRESH):
            return True  # not vertical enough
        else:
            return False

    def off_table(self, pos):
        if pos[0]<self.table_min_x or pos[0]>self.table_max_x:
            #print('off table in x')
            return True
        if pos[1]<self.table_min_y or pos[1]>self.table_max_y:
            #print('off table in y')
            return True
        return False

    def get_all_init_object_poses(self):
        assert(False)  # subclasses should implement

    def get_init_pos(self):
        assert(False)  # subclasses should implement
