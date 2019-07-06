#
# Torque control with minimum jerk EE waypoints.
#
import numpy as np
import math
import pybullet

# Conforming to pybullet's conventions on reporting angular velocity.
# Different from pybullet.GetQuaternionFromEuler().
def quaternion_to_euler(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.pi/2.0
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([pitch, -roll, yaw])


def euler_to_quaternion(theta):
    """
    Assuming that theta is from the quaternion to euler function above
    :param theta:
    :return:
    """

    roll = -theta[1]
    pitch = theta[0]
    yaw = theta[2]

    quat = pybullet.getQuaternionFromEuler([roll, pitch, yaw])
    return quat


def plan_linear_orientation_trajectory(th0, th_goal, N):
   shortest_path = (((th_goal-th0) + np.pi) % (2.0*np.pi)) - np.pi
   amount = np.linspace(0.0, 1.0, num=N)
   traj = np.zeros((N, th0.shape[0]))
   for i in range(N):
       traj[i] = th0 + (shortest_path*amount[i])
   return traj


def plan_min_jerk_trajectory(y0, goal, dur, dt):
    N = int(dur / dt)
    nDim = np.shape(y0)[0]
    Y = np.zeros((N, nDim))
    Yd = np.zeros((N, nDim))
    Ydd = np.zeros((N, nDim))
    Y[0, :] = y0
    rem_dur = dur
    for n in range(1, N):
        y_curr = Y[n - 1, :]
        yd_curr = Yd[n - 1, :]
        ydd_curr = Ydd[n - 1, :]
        for d in range(nDim):
            Y[n, d], Yd[n, d], Ydd[n, d] = calculate_min_jerk_step(
                y_curr[d], yd_curr[d], ydd_curr[d], goal[d], rem_dur, dt)
        rem_dur = rem_dur - dt
    return Y, Yd, Ydd

def calculate_min_jerk_step(y_curr, yd_curr, ydd_curr, goal, rem_dur, dt):

    if rem_dur < 0:
        return

    if dt > rem_dur:
        dt = rem_dur

    t1 = dt
    t2 = t1 * dt
    t3 = t2 * dt
    t4 = t3 * dt
    t5 = t4 * dt

    dur1 = rem_dur
    dur2 = dur1 * rem_dur
    dur3 = dur2 * rem_dur
    dur4 = dur3 * rem_dur
    dur5 = dur4 * rem_dur

    dist = goal - y_curr
    a1t2 = 0.0  # goaldd
    a0t2 = ydd_curr * dur2
    v1t1 = 0.0  # goald
    v0t1 = yd_curr * dur1

    c1 = (6.0 * dist + (a1t2 - a0t2) / 2.0 - 3.0 * (v0t1 + v1t1)) / dur5
    c2 = (-15.0 * dist + (3.0 * a0t2 - 2.0 * a1t2) /
          2.0 + 8.0 * v0t1 + 7.0 * v1t1) / dur4
    c3 = (10.0 * dist + (a1t2 - 3.0 * a0t2) /
          2.0 - 6.0 * v0t1 - 4.0 * v1t1) / dur3
    c4 = ydd_curr / 2.0
    c5 = yd_curr
    c6 = y_curr

    y = c1 * t5 + c2 * t4 + c3 * t3 + c4 * t2 + c5 * t1 + c6
    yd = 5 * c1 * t4 + 4 * c2 * t3 + 3 * c3 * t2 + 2 * c4 * t1 + c5
    ydd = 20 * c1 * t3 + 12 * c2 * t2 + 6 * c3 * t1 + 2 * c4

    return y, yd, ydd
