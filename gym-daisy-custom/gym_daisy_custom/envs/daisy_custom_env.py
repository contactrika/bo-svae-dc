#
# A customized env for Daisy robot simulations.
#
# @contactrika
#
from copy import copy

import numpy as np

from gym import spaces
import pybullet

from daisy_env.daisy_environments import DaisyBasicEnv
import daisy_env.configs as DaisyConfigs


class DaisyCustomEnv(DaisyBasicEnv):
    MAX_X = 1.0  # width of the corridor/room
    MIN_Z = 0.1  # minimum ok height for the base
    MAX_Z = 0.75 # minimum ok height for the base
    MAX_JOINT_VEL_SIM = 5.0  # maximum joint velocity we want to allow
    MAX_JOINT_ANGLE = np.pi
    MAX_BASE_POS = np.array([10.0, 10.0, 10.0])
    # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12077
    MAX_VELOCITY = 100.0  # 100 rad/s
    # Limiting this so that we have somewhat reasonable bullet simulation.
    MAX_TORQUE = 10000.0
    # Targets and obstacles.
    MARKERS_POS = [[0, 0, 0.02], [0, 2.0, 0.05]]
    MARKERS_RADIUS = [0.02, 0.05]
    MARKERS_RGBA = [[0, 0, 1, 1], [1, 0.5, 0, 1]]  # blue, orange
    # From DaisyConfigs.fast_standing_6legged_config
    RESTITUTION = 0.05
    LATERAL_FRICITON = 2.0
    # Default pybullet for link linear and angular damping is 0.04
    # Join damping for Daisy is 0.5. Changing joint damping doesn't seem to
    # affect the simulation much (unless it is set to values >50); increasing
    # link damping seems to make the robot more stiff and make it move a bit
    # more awkward and slower, matching rea-world motion a little better.
    DAMPING = 0.1

    def __init__(self, max_episode_steps=300, control_mode='position',
                 can_terminate_early=False, penalize_stalled=True,
                 visualize=False, debug_level=0):
        self.pos_only_state = True  # don't include velocities in state
        self.penalize_stalled = penalize_stalled # for sim, not used for hw
        self.debug_level = debug_level
        self.max_episode_steps = max_episode_steps
        self.can_terminate_early = can_terminate_early
        self.visualize = visualize
        cfg = DaisyConfigs.fast_standing_6legged_config
        cfg['render'] = visualize
        cfg['self_collision'] = True
        self.sim_step = cfg['sim_timestep']
        super(DaisyCustomEnv, self).__init__(**cfg)
        self.robot.set_control_mode(control_mode)
        self._make_markers()
        obs, _, _ = super(DaisyCustomEnv, self)._daisy_base_reset()
        max_poses = np.ones_like(obs['j_pos'])*DaisyCustomEnv.MAX_JOINT_ANGLE
        self.num_joints = max_poses.shape[0]
        max_base_pos = DaisyCustomEnv.MAX_BASE_POS
        max_sincos = np.ones([6], dtype=float)  # euler angles sin,cos
        max_vels = np.ones_like(obs['j_vel'])*DaisyCustomEnv.MAX_VELOCITY
        max_base_vel = np.array([DaisyCustomEnv.MAX_VELOCITY]*3)  # vel vec
        if self.pos_only_state:
            self.obs_highs = np.hstack([max_poses, max_base_pos, max_sincos])
            dim_wts = np.hstack(
                [np.repeat(1.0/self.num_joints, self.num_joints),
                 DaisyCustomEnv.MAX_BASE_POS*np.repeat(1.0/3, 3),
                 np.repeat(1.0/6, 6)])
        else:
            self.obs_highs = np.hstack([
                max_poses, max_vels, max_base_pos, max_sincos, max_base_vel])
            dim_wts = np.hstack([
                np.repeat(1.0/self.num_joints, self.num_joints),
                np.repeat(1.0/self.num_joints, self.num_joints),
                DaisyCustomEnv.MAX_BASE_POS*np.repeat(1.0/3, 3),
                np.repeat(1.0/6, 6), np.repeat(1.0/3, 3)])
        # Make dim_weights sum to number of steps (the original sum)
        self.dim_weights = dim_wts/np.sum(np.abs((dim_wts)))*dim_wts.shape[0]
        self.obs_lows = -self.obs_highs
        self.observation_space = spaces.Box(self.obs_lows, self.obs_highs)
        if control_mode == 'position':
            self.action_space = spaces.Box(
                np.array([-1.0*DaisyCustomEnv.MAX_JOINT_ANGLE]*self.num_joints),
                np.array([DaisyCustomEnv.MAX_JOINT_ANGLE]*self.num_joints))
        elif control_mode == 'velocity':
            self.action_space = spaces.Box(
                np.array([-1.0*DaisyCustomEnv.MAX_VELOCITY]*self.num_joints),
                np.array([DaisyCustomEnv.MAX_VELOCITY]*self.num_joints))
        else:
            print('Unsupported control_mode', control_mode)
            assert(False)
        # Create a list of names for obs for easier debugging.
        obs_names = ['pos_joint'+str(i) for i in range(max_poses.shape[0])]
        if not self.pos_only_state:
            obs_names += ['vel_joint'+str(i) for i in range(max_vels.shape[0])]
        obs_names.extend([
            'base_pos_x', 'base_pos_y', 'base_pos_z',
            'base_pos_sin0', 'base_pos_sin1', 'base_pos_sin2',
            'base_pos_cos0', 'base_pos_cos1', 'base_pos_cos2'])
        if not self.pos_only_state:
            obs_names += ['base_vel'+str(i)
                          for i in range(max_base_vel.shape[0])]
        self.obs_names = ['daisy_'+nm for nm in obs_names]

    def get_obs_names(self):
        return self.obs_names

    def _make_markers(self):
        for mrkid in range(len(DaisyCustomEnv.MARKERS_RADIUS)):
            viz_shp_id = self._p.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE, visualFramePosition=[0,0,0],
                radius=DaisyCustomEnv.MARKERS_RADIUS[mrkid],
                rgbaColor=DaisyCustomEnv.MARKERS_RGBA[mrkid])
            col_shp_id = self._p.createCollisionShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=DaisyCustomEnv.MARKERS_RADIUS[mrkid],
                collisionFramePosition=[0,0,0])
            # Only using this for visualization, so mass=0 (fixed body).
            self.origin_body_id = self._p.createMultiBody(
                baseMass=0, baseInertialFramePosition=[0,0,0],
                baseCollisionShapeIndex=col_shp_id,
                baseVisualShapeIndex=viz_shp_id,
                basePosition=DaisyCustomEnv.MARKERS_POS[mrkid],
                useMaximalCoordinates=True)

    def state_from_state_dict(self, state_dict):
        if (np.abs(state_dict['j_pos'])>DaisyCustomEnv.MAX_JOINT_ANGLE).any():
            print('fixing j_pos', state_dict['j_pos'])
            state_dict['j_pos'] = np.clip(
                state_dict['j_pos'], -DaisyCustomEnv.MAX_JOINT_ANGLE,
                DaisyCustomEnv.MAX_JOINT_ANGLE)
        assert(np.abs(state_dict['j_pos'])
               <= DaisyCustomEnv.MAX_JOINT_ANGLE).all()
        #base_quat = state_dict['base_ori_quat']
        #base_rot = np.array(pybullet.getMatrixFromQuaternion(base_quat))
        #assert((base_rot>=-1.0).all()); assert((base_rot<=1.0).all())
        # Using cos and sin of euler angles to remove discontinuities.
        base_euler = state_dict['base_ori_euler']
        base_sin = np.sin(base_euler); base_cos = np.cos(base_euler)
        base_pos = np.hstack([state_dict['base_pos_x'],
                              state_dict['base_pos_y'],
                              state_dict['base_pos_z']])
        base_pos = np.clip(base_pos, -DaisyCustomEnv.MAX_BASE_POS,
                           DaisyCustomEnv.MAX_BASE_POS)
        if self.pos_only_state:
            state = np.hstack([state_dict['j_pos'],
                               base_pos, base_sin, base_cos])
        else:
            assert(np.abs(state_dict['j_vel'])
                   <= DaisyCustomEnv.MAX_VELOCITY).all()
            assert((state_dict['base_velocity']
                    <= DaisyCustomEnv.MAX_VELOCITY).all())
            state = np.hstack([state_dict['j_pos'], state_dict['j_vel'],
                               base_pos, base_sin, base_cos,
                               state_dict['base_velocity']])
        assert((state>=self.obs_lows).all())
        assert((state<=self.obs_highs).all())
        return state

    def state_dict_from_state(self, state):
        state_dict = {}
        state_dict['j_pos'] = state[0:self.num_joints]
        sid = self.num_joints
        if not self.pos_only_state:
            state_dict['j_vel'] = state[sid:sid+self.num_joints]
            sid += self.num_joints
        state_dict['base_pos_x'] = state[sid]
        state_dict['base_pos_y'] = state[sid+1]
        state_dict['base_pos_z'] = state[sid+2]
        sid += 3  # 3D base pos
        base_sin = state[sid:sid+3]; base_cos = state[sid+3:sid+6]
        state_dict['base_ori_euler'] = np.arctan2(base_sin, base_cos)
        state_dict['base_ori_quat'] = pybullet.getQuaternionFromEuler(
            state_dict['base_ori_euler'])
        sid += 6
        if not self.pos_only_state: state_dict['base_velocity'] = state[sid:]
        return state_dict

    def reset(self):
        if self.visualize: input("Press Enter to continue env reset...")
        self.stepnum = 0
        self.episode_reward = 0
        self.badlen = 0
        state_dict, _, _ = super(DaisyCustomEnv, self)._daisy_base_reset()
        return self.state_from_state_dict(state_dict)

    def randomize(self, debug=False):
        min_scale = 0.8; max_scale = 1.2
        scalings = np.random.rand(3)*(max_scale-min_scale) + min_scale
        restitution = DaisyCustomEnv.RESTITUTION*scalings[0]
        lateral_friction = DaisyCustomEnv.LATERAL_FRICITON*scalings[1]
        damping = DaisyCustomEnv.DAMPING*scalings[2]
        rnd_params = np.array([restitution, lateral_friction, damping])
        self.set_randomize(rnd_params, debug)
        return rnd_params

    def set_randomize(self, rnd_params, debug=False):
        assert(rnd_params.shape[0]==3)
        restitution = rnd_params[0]
        lat_fric = rnd_params[1]
        damping = rnd_params[2]
        #self.robot.set_friction_and_restitution(
        #    self._p, lateralFriction=lateral_friction, restitution=restitution)
        body_idx = self.robot.robot_body.bodies[0]
        bullet_client = self.robot._p
        numLinks = bullet_client.getNumJoints(body_idx)
        bullet_client.changeDynamics(
            body_idx, -1, restitution=restitution, lateralFriction=lat_fric)
        for joint_idx in range(numLinks):
            bullet_client.changeDynamics(
                body_idx, joint_idx, restitution=restitution,
                lateralFriction=lat_fric, jointDamping=damping,
                linearDamping=damping, angularDamping=damping)
        if debug:
            msg = 'randomize(): restitution {:0.4f} lateral friction {:0.4f}'
            msg += ' damping {:0.4f}'
            print(msg.format(restitution, lat_fric, damping))

    def step(self, action):
        prev_state_dict = self.robot.calc_state()
        if self.debug_level>=5:
            print('----------- step {:d}--------------'.format(self.stepnum))
        if isinstance(action, np.ndarray): action = action.tolist()
        #if self._do_render:
        #self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        self.robot.apply_action(action)
        self.scene.global_step()
        #self.camera_adjust()
        #if self._do_render:
        #self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
        next_state_dict = self.robot.calc_state()
        self._alive = 1.0  # from original code (not sure if needed here)
        reward = self.get_reward_dense(prev_state_dict, next_state_dict)
        is_bad = self.get_is_bad(prev_state_dict, next_state_dict)
        self.badlen += is_bad
        self.stepnum += 1
        done = (self.stepnum == self.max_episode_steps)
        if self.can_terminate_early:
            terminate, reward_adjust = self.should_terminate(next_state_dict)
            if terminate:
                reward += reward_adjust
                done = True
        self.episode_reward += reward  # needs to happen after reward_adjust
        # Return results.
        info = {'is_bad': is_bad}
        next_state = self.state_from_state_dict(next_state_dict)
        if done:
            bad_frac = self.badlen/self.stepnum
            info['done_reward'] = self.episode_reward
            info['done_badfrac'] = bad_frac
            info['done_obs'] = copy(next_state) # vector envs lose last frame
            if self.debug_level>=1:
                print('Final base_pos_y', next_state_dict['base_pos_y'])
                print('done_reward {:0.4f} badlen {:d}'.format(
                    info['done_reward'], self.badlen))
        # From original code (not sure if needed here)
        for obj in self._environment_hooks:
            obj.step(self, next_state_dict, 0.0, bool(self._isDone()))

        return next_state, reward, done, info

    def should_terminate(self, state_dict):
        # Stop episode if went outside the boundaries; this can be used
        # on hardware where there is little space in the room.
        x = state_dict['base_pos_x']
        if np.abs(x) > DaisyCustomEnv.MAX_X:
            if self.debug_level>=0: print('Went outside at step', self.stepnum)
            return True, 0.0  # not penalizing going toward walls
        rpy = state_dict['base_ori_euler']
        if np.any(np.absolute(rpy[0:2]) > np.pi/3):
            if self.debug_level>=0: print('Almost tipped over rpy', rpy)
            return True, -50.0  # penalizing for tipping over

        return False, 0.0  # everything ok

    def get_reward_dense(self, prev_state_dict, state_dict):
        y = float(state_dict['base_pos_y'])
        prev_y = float(prev_state_dict['base_pos_y'])
        y_reward = float(y - prev_y)*10.0            # good if moved forward
        j_pos = np.array(state_dict['j_pos'])
        prev_j_pos = np.array(prev_state_dict['j_pos'])
        vel = (j_pos-prev_j_pos)/self.sim_step
        vel_reward = 0.0
        if (np.abs(vel)>DaisyCustomEnv.MAX_JOINT_VEL_SIM).any():
            vel_reward = -1.0  # high joint vels are bad
        rwd = y_reward + vel_reward
        return float(rwd)  # float to make a scalar

    def get_is_bad(self, prev_state_dict, state_dict, debug=False):
        x = float(state_dict['base_pos_x'])
        y = float(state_dict['base_pos_y'])
        z = float(state_dict['base_pos_z'])
        if x<-1.0*DaisyCustomEnv.MAX_X or x>DaisyCustomEnv.MAX_X:
            if debug: print('problematic x', x)
            return True
        if z<DaisyCustomEnv.MIN_Z or z>DaisyCustomEnv.MAX_Z:
            if debug: print('problematic z', z)
            return True
        prev_j_pos = np.array(prev_state_dict['j_pos'])
        j_pos = np.array(state_dict['j_pos'])
        if ((j_pos.reshape(-1,3)[0:3,2]>0).any() or
            (j_pos.reshape(-1,3)[3:6,2]<0).any()):
            if debug: print('elbows out jpos', j_pos)
            return True
        vel = (j_pos-prev_j_pos)/self.sim_step
        if np.abs(vel).any() > DaisyCustomEnv.MAX_JOINT_VEL_SIM:
            if debug: print('problematic vel', vel)
            return True
        rpy = state_dict['base_ori_euler']
        if np.any(np.absolute(rpy) > np.pi/4):
            if debug: print('problematic rpy', rpy)
            return True
        if self.penalize_stalled:
            prev_x = float(prev_state_dict['base_pos_x'])
            prev_y = float(prev_state_dict['base_pos_y'])
            v0 = np.array([x,y]); v1 = np.array([prev_x,prev_y])
            moved = np.linalg.norm(v1-v0)/self.sim_step
            if moved < 0.05:
                if debug: print('stalled: moved only', moved)
                return True
        if debug: print('get_is_bad: no!')
        return False  # everything is ok if we got here

    def render(self, mode="rgb_array", close=False):
        # Implemented in pybullet
        pass

    def camera_adjust(self):
        # pybullet.resetDebugVisualizerCamera(
        #     cameraDistance=3, cameraYaw=30, cameraPitch=52,
        #     cameraTargetPosition=[0,0,0])
        #if self._p is None or not self._do_render: return
        assert(self._p is not None)
        self.camera._p = self._p
        #camera_tgt = self.robot.body_xyz
        camera_tgt = [0,0,0]  # turn off camera motion
        self._p.resetDebugVisualizerCamera(
            cameraDistance=2.0, cameraYaw=30, cameraPitch=-35,
            cameraTargetPosition=camera_tgt)

    def render_debug(self, mode="rgb_array", close=False, width=600):
        #self.bullet_env.sim.resetDebugVisualizerCamera(
        #    cameraDistance=1, cameraYaw=-30, cameraPitch=-30,
        #    cameraTargetPosition=[0, 0, 0])
        # Does not render obstacles even with
        # renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12016
        if mode != "rgb_array": return np.array([])
        height = width
        #camera_tgt = self.robot.body_xyz
        camera_tgt = [0,0,0]  # turn off camera motion
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_tgt, distance=1.4,
            yaw=0, pitch=-45, roll=0, upAxisIndex=2)
        proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov=90, aspect=float(width)/height,
            nearVal=0.01, farVal=100.0)
        w, h, rgba_px, depth_px, segment_mask = self._p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER)
        #import scipy.misc
        #scipy.misc.imsave('/tmp/outfile.png', rgba_px)
        return rgba_px  # HxWxRGBA uint8

    def render_debug_other(self, width=600):
        if width is not None or self._projectM is None:
            if self._projectM is None:
                self._pixelWidth = self._param_init_camera_width
                self._pixelHeight = self._param_init_camera_height
            else:
                self._pixelWidth = width
                self._pixelHeight = width
            nearPlane = 0.01
            farPlane = 10
            aspect = self._pixelWidth / self._pixelHeight
            fov = 60
            self._projectM = self._p.computeProjectionMatrixFOV(fov, aspect,
                nearPlane, farPlane)
        x, y, z = self.robot.body_xyz
        lookat = [x, y, z]
        distance = 1.4
        yaw = -20
        viewM = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=lookat, distance=distance,
            yaw=10., pitch=yaw, roll=0.0, upAxisIndex=2)
        img_arr = self._p.getCameraImage(
            self._pixelWidth, self._pixelHeight, viewM, self._projectM,
            shadow=self._param_COV_ENABLE_SHADOWS,
            renderer=pybullet.ER_TINY_RENDERER,  #ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_NO_SEGMENTATION_MASK)
        return img_arr[2] #color data RGBA

    def override_state(self, state):
        state_dict = self.state_dict_from_state(state)
        base_pos = [state_dict['base_pos_x'],
                    state_dict['base_pos_y'],
                    state_dict['base_pos_z']]
        self.robot.parts['daisy'].reset_pose(
            base_pos, state_dict['base_ori_quat'])
        if not self.pos_only_state:
            self.robot.parts['daisy'].reset_velocity(
                state_dict['base_velocity'])
        for idx, joint in enumerate(self.robot.ordered_joints):
            #print('joint name', joint.joint_name)
            if self.pos_only_state:
                joint.reset_position(state[idx], 0.0)
            else:
                joint.reset_position(state[idx], state[self.num_joints+idx])
