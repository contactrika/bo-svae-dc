#
# Parametric gait controllers.
#
import numpy as np

from .cpg_oscillator import CpgController


class Robot:
    def __init__(self, n_legs, n_joint_per_leg):
        self.n_legs = n_legs
        self.n_joint_per_leg = n_joint_per_leg
        # As many oscillators as joints. Amplitude of knee set to 0
        self.n_oscillators = self.n_legs*n_joint_per_leg


class Gait:
    def __init__(self, robot, relative_phase, v, a, R, amp_offset, phase_offset):
        """
        Class containing default components of a gait. Formulation similar to Crespi2008
        :param robot: Robot object
        :param relative_phase: phase difference between base joints
        :param v: frequency of each CPG - fixed and same right now
        :param R: amplitude of CPG - fixed and same right now
        :param a: positive constant
        :param phase_offset: Phase difference between first-second and second-third joint
        """

        self.robot = robot
        self.relative_phase = relative_phase
        self.v = v
        self.a = a
        self.R = R
        self.amp_offset = amp_offset
        self.phase_offset = phase_offset

        self.phase_biases = self.generate_biases(relative_phase)

    def generate_biases(self, relative_phase):
        """
        Generates the bias matrix depending on the gait.
        relative_phase defines the phase difference between the 6 base joints

        """
        phase_biases = np.zeros((self.robot.n_oscillators, self.robot.n_oscillators))  # All connections
        #  initialized to zero
        # phase offset between the first and second joint - row1
        # phase offset between base and third joint - row2
        # phase offset between base joints - relative phases
        for i in range(self.robot.n_legs):
            #import pdb; pdb.set_trace()
            phase_biases[i][i + self.robot.n_legs] = -self.phase_offset[0][i]
            phase_biases[i][i + 2*self.robot.n_legs] = -self.phase_offset[1][i]
            phase_biases[i + self.robot.n_legs][i] = self.phase_offset[0][i]
            phase_biases[i + 2*self.robot.n_legs][i] = self.phase_offset[1][i]
        # phase offset between base joints, depends on gait
            for j in range(self.robot.n_legs):
                phase_biases[i][j] = relative_phase[i][j]

        return phase_biases


"""
GAIT parameters
Relative phase : 6-by-6 matrix, skew symmetric, phase diff between base joints
v : frequency of each oscillator
a : constant for each oscillator
R : amplitude
amp_offset: constant offset in each oscillator
phase_offset: offset between base-shoulder and base-elbow joints
Coupling weights are constant right now - but can be added and changed in CpgController class
"""
class DaisyGaitConstants:
    ACTION_DIM = 6*3
    ROBOT = Robot(n_legs=6, n_joint_per_leg=3)
    TRIPOD_EXAMPLE = Gait(
        robot=ROBOT,
        relative_phase=[[0,      np.pi,      0,      0,      0,      np.pi],
                        [-np.pi, 0,          np.pi,  0,      0,      0],
                        [0,      -np.pi,     0,      np.pi,  0,      0],
                        [0,      0,          -np.pi, 0,      np.pi,  0],
                        [0,      0,          0,      -np.pi, 0,      np.pi],
                        [-np.pi, 0,          0,      0,      -np.pi, 0]],
        v=[[2 for _ in range(6)], [2 for _ in range(6)], [2 for _ in range(6)]],
        a=[[0.5 for _ in range(6)], [0.5 for _ in range(6)], [0.5 for _ in range(6)]],
        R=[[0.15 for _ in range(6)], [0.15 for _ in range(6)], [0.15 for _ in range(6)]],
        amp_offset=[[0.15 for _ in range(6)], [0.15 for _ in range(6)], [0.15 for _ in range(6)]],
        phase_offset=[[np.pi/2 for _ in range(6)], [np.pi/2 for _ in range(6)]]
    )


class DaisyGait27DPolicy:
    DIM = 27  # 27-dimensional controller

    def __init__(self, x, traj_len, sim, init_pos_fxn=None):
        # x: controller params scaled in [0,1] (numpy array)
        # This code is from run_controller_return_traj.py
        # TODO: finish refactoring and comments
        self.gait = DaisyGait27DPolicy.make_gait(x)
        self.cpg, self.actions = DaisyGait27DPolicy.make_cpg(
            self.gait, traj_len)

    @staticmethod
    def make_cpg(gait, traj_len):
        cpg = CpgController(gait)  # open-loop control (obs not used)
        actions = np.zeros(
            [traj_len, DaisyGaitConstants.ACTION_DIM],dtype=float)
        act = np.zeros(DaisyGaitConstants.ACTION_DIM, dtype=float)
        for i in range(traj_len):  # actions for the whole trajectory
            cpg.update()
            t = 0
            for j in range(cpg.n_legs):
                if np.remainder(j, 2) == 0:
                    p = int(j/2)
                    act[t] = cpg.y_data[p][i]
                    act[t+1] = cpg.y_data[cpg.n_legs+p][i]
                    act[t+2] = -1 + cpg.y_data[2*cpg.n_legs+p][i]
                else:
                    p = int(np.floor(j/2))
                    act[t] = -cpg.y_data[cpg.n_legs-p-1][i]
                    act[t+1] = -cpg.y_data[2*cpg.n_legs-p-1][i]
                    act[t+2] = 1 - cpg.y_data[3*cpg.n_legs-p-1][i]
                t += 3
            actions[i, :] = act[:]
        return cpg, actions

    @staticmethod
    def make_gait(x):
        phases = x[0:6]*2*np.pi - np.pi  # Between -pi to pi

        relative_phases = [[0 for _ in range(6)] for _ in range(6)]
        for i in range(6):
            for j in range(6):
                relative_phases[i][j] = phases[j] - phases[i]

        phase_offset = np.vstack([x[6:12], x[12:18]])*2*np.pi - np.pi
        # phase_offset = np.tile(np.asarray(phases), 6)

        # v = [x[18:24], x[24:30], x[30:36]]
        vs = np.vstack([x[18], x[19], x[20]]) * 2
        v = np.tile(np.asarray(vs), 6)

        # amp = [x[36:42], x[42:48], x[48:54]]
        amps = np.vstack([x[21], x[22], x[23]])
        amp = np.tile(np.asarray(amps), 6)

        # offset = [x[54:60], x[60:66], x[66:]]
        offs = np.vstack([x[24], x[25], x[26]])
        offset = np.tile(np.asarray(offs), 6)

        a = np.array([[1, 1, 1]])
        a = np.repeat(np.asarray(a), 6, axis=0).transpose()

        return Gait(
            robot=DaisyGaitConstants.ROBOT,
            relative_phase=relative_phases,
            v=v,
            a=a,
            R=amp,
            amp_offset=offset,
            phase_offset=phase_offset)

    def get_action(self, obs, t):
        return self.actions[t]


class DaisyGait11DPolicy:
    DIM = 11  # 11-dimensional controller

    def __init__(self, x, traj_len, sim, init_pos_fxn=None):
        # x: controller params scaled in [0,1] (numpy array)
        # This code is from run_controller_return_traj.py
        # TODO: finish refactoring and comments
        self.gait = DaisyGait11DPolicy.make_gait(x)
        self.cpg, self.actions = DaisyGait11DPolicy.make_cpg(
            self.gait, traj_len)

    @staticmethod
    def make_cpg(gait, traj_len):
        cpg = CpgController(gait)  # open-loop control (obs not used)
        actions = np.zeros(
            [traj_len, DaisyGaitConstants.ACTION_DIM],dtype=float)
        act = np.zeros(DaisyGaitConstants.ACTION_DIM, dtype=float)
        for i in range(traj_len):  # actions for the whole trajectory
            cpg.update()
            t = 0
            # 11D controller with further joint limits to make it safer for
            # hardware experiments.
            # From daisy-bo/test_sample_gait.py#L104 test_sample_gait_hdw()
            for j in range(cpg.n_legs):
                if np.remainder(j, 2) == 0:
                    p = int(j / 2)
                    act[t] = -max(min(cpg.y_data[p][i], 0.6), -0.6)
                    act[t + 1] = -max(min(cpg.y_data[cpg.n_legs+p][i], 1.0), -1.0)
                    act[t + 2] = max(min(-1 + cpg.y_data[2*cpg.n_legs+p][i], 1.5), -1.5)
                else:
                    p = int(np.floor(j / 2))
                    act[t] = -max(min(-cpg.y_data[cpg.n_legs-p-1][i], 0.6), -0.6)
                    act[t + 1] = -max(min(-cpg.y_data[2 * cpg.n_legs-p-1][i], 1.0), -1.0)
                    act[t + 2] = max(min(1 - cpg.y_data[3 * cpg.n_legs-p-1][i], 1.5), -1.5)
                t += 3
            actions[i, :] = act[:]
        return cpg, actions

    @staticmethod
    def make_gait(x):
        phases = x[0:6] * 2 * np.pi - np.pi  # Between -pi to pi

        relative_phases = [[0 for _ in range(6)] for _ in range(6)]
        for i in range(6):
            for j in range(6):
                relative_phases[i][j] = phases[j] - phases[i]

        phases = np.vstack([x[6], x[7]]) * 2 * np.pi - np.pi
        phase_offset = np.tile(np.asarray(phases), 6)

        # v = [x[18:24], x[24:30], x[30:36]]
        vs = np.vstack([x[8], x[8], x[8]]) * 0.5 + 0.5
        v = np.tile(np.asarray(vs), 6)

        # amp = [x[36:42], x[42:48], x[48:54]]
        amps = np.vstack([x[9], x[9], x[9]]) * 0.5 + 0.5
        amp = np.tile(np.asarray(amps), 6)

        # offset = [x[54:60], x[60:66], x[66:]]
        offs = np.vstack([x[10], x[10], x[10]]) * 0.5 + 0.5
        offset = np.tile(np.asarray(offs), 6)

        a = np.array([[1, 1, 1]])
        a = np.repeat(np.asarray(a), 6, axis=0).transpose()

        return Gait(
            robot=DaisyGaitConstants.ROBOT,
            relative_phase=relative_phases,
            v=v,
            a=a,
            R=amp,
            amp_offset=offset,
            phase_offset=phase_offset)

    def get_action(self, obs, t):
        return self.actions[t]


class DaisyTripod27DPolicy:
    DIM = 27  # 27-dimensional controller

    def __init__(self, x, traj_len, sim, init_pos_fxn=None):
        self.gait = DaisyGaitConstants.TRIPOD_EXAMPLE
        self.cpg, self.actions = DaisyGait27DPolicy.make_cpg(self.gait, traj_len)

    def get_action(self, obs, t):
        return self.actions[t]
