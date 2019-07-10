#
# Oscillator code for CPG controller.
#
# @akshara
#
import numpy as np
import matplotlib.pyplot as plt
import pdb

WEIGHT = 10
STEP_SIZE = 0.01


class Oscillator:
    def __init__(self, v, a, R, off):
        """
        Each oscillator in the CPG network
        :param v: frequency
        :param a: constant
        :param R: amplitude
        """
        self.v = v
        self.a = a
        self.R = R

        self.phase = 0
        self.r = R
        self.dr = 0
        self.offset = off

    def output(self):
        return self.r*(1 + np.cos(self.phase)) - self.offset


class OscillatorNetwork:
    def __init__(self, oscillators, weights, phase_biases):
        """
        Network of oscillators. Implements the evolution of the CPGs

        :param oscillators: Sequence of oscillators
        :param weights: the weights that couple the oscillators
        :param phase_biases: phase biases between the oscillators
        """
        self.oscillators = oscillators
        self.weights = weights
        self.phase_biases = phase_biases
        self.n_oscillators = len(oscillators)

    def update_all(self):
        """
        One step of integration of all the oscillators
        :return:
        """
        for i in range(self.n_oscillators):
            current = self.oscillators[i]
            phase_change = 2*np.pi*current.v
            for j in range(self.n_oscillators):
                other = self.oscillators[j]
                phase_change += self.weights[i][j]*np.sin(other.phase - current.phase - self.phase_biases[i][j])
            self.oscillators[i].phase += phase_change*STEP_SIZE

            a = current.a
            ddr = a*(0.25*a*(current.R - current.r) - current.dr)
            self.oscillators[i].dr += ddr*STEP_SIZE
            self.oscillators[i].r += self.oscillators[i].dr*STEP_SIZE
            self.oscillators[i].x = self.oscillators[i].output()


class CpgController:
    def __init__(self, gait):
        """
        CPG controller for Daisy. Based on the controller from Crespi2008
        Phase difference between all base joints defined in phase_grouping
        Phase difference between base and first joint in phase_offset
        Phase difference between first and second joint in each leg is 0

        :param gait: Gait object from cpg_gait_robot
        """
        self.gait = gait
        self.robot = gait.robot
        self.n_legs = self.robot.n_legs
        self.n_joint_per_leg = self.robot.n_joint_per_leg
        self.n_oscillators = self.robot.n_oscillators

        # Initialize oscillators
        self.oscillators = []
        T = []
        for joint in range(self.n_joint_per_leg):
            T.append([Oscillator(v=gait.v[joint][_], a=gait.a[joint][_], R=gait.R[joint][_], off=gait.amp_offset[joint][_]) for _ in range(self.robot.n_legs)])
            self.oscillators += T[joint]

        # Create weights - 0s everywhere except in coupled joints
        # Currently third joint is not coupled with the base and second joint
        self.coupling_weights = np.zeros((self.robot.n_oscillators, self.robot.n_oscillators))
        for i in range(self.robot.n_legs):
            self.coupling_weights[i][i + self.n_legs] = WEIGHT
            self.coupling_weights[i + self.n_legs][i] = WEIGHT
            self.coupling_weights[i][i + 2*self.n_legs] = WEIGHT
            self.coupling_weights[i + 2*self.n_legs][i] = WEIGHT
            for j in range(self.n_legs):
                self.coupling_weights[i][j] = WEIGHT

        self.phase_biases = self.gait.phase_biases
        self.network = OscillatorNetwork(self.oscillators, self.coupling_weights, self.phase_biases)

        # Plotting tools
        self.time = 0
        self.x_data = []
        self.y_data = [[] for _ in range(self.n_oscillators)]

    def update(self, logging=True):
        self.network.update_all()
        if logging:
            self.x_data.append(self.time)
            self.time += STEP_SIZE

            output = self.output()
            for i in range(self.n_oscillators):
                self.y_data[i].append(output[i])

    def output(self):
        return [self.oscillators[i].x for i in range(self.n_oscillators)]

    def plot(self):
        for y in self.y_data:
            plt.plot(self.x_data, y, linewidth=2.5)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Oscillator output', fontsize=12)
        plt.show()
