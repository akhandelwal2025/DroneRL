from utils import *
import config
from Drone import Drone
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import casadi as ca
import numpy as np

def mpc_predict_future_n_steps(x0, u0, x_target, n_steps, dt=0.1, optimizer='ipopt'):
    assert optimizer in ['ipopt', 'qpoases'], "only supported optimizers are ipopt and qpoases"

    # Define symbolic state and control variables
    x   = ca.SX.sym('x',3)     # Position
    th  = ca.SX.sym('th',3)    # Orientation (Euler angles)
    v   = ca.SX.sym('v',3)     # Linear velocity
    omg = ca.SX.sym('omg',3)   # Angular velocity
    u   = ca.SX.sym('u',4)     # Control inputs (normalized motor thrusts in [0, 1])

    # Drone parameters
    m   = 0.468                              # Mass in kg
    I   = ca.diag(ca.SX([4.856e-3, 4.856e-3, 8.801e-3]))  # Inertia matrix
    kf, kd = 2.98e-6, 1.14e-7                # Lift and drag coefficients
    force_to_torque = kd / kf               # Torque coefficient per thrust unit
    l   = 0.225                              # Arm length (from center to rotor)
    max_thrust = 10                          # Maximum thrust per motor (N)

    # Rotation matrix from body frame to world frame
    cx, cy, cz = ca.cos(th[0]), ca.cos(th[1]), ca.cos(th[2])
    sx, sy, sz = ca.sin(th[0]), ca.sin(th[1]), ca.sin(th[2])
    R = ca.vertcat(
        ca.horzcat(cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx),
        ca.horzcat(sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx),
        ca.horzcat(-sy   , cy*sx           , cy*cx)
    )

    # Compute thrust vector in body frame and convert to world frame
    T = u * max_thrust                              # Actual thrust in N
    acc_b = ca.vertcat(0, 0, ca.sum1(T)) / m        # Body-frame acceleration (along z-axis)
    acc_w = R @ acc_b - ca.vertcat(0, 0, 9.8)       # World-frame acceleration (gravity subtracted)
    # acc_w = R @ acc_b

    # Compute torques from each motor's position and thrust
    fr_theta = math.radians(45)
    fl_theta = math.radians(135)
    rl_theta = math.radians(225)
    rr_theta = math.radians(315)
    
    tau_fl = ca.cross(ca.vertcat(math.cos(fl_theta), math.sin(fl_theta), 0)*l, T[0]*ca.vertcat(0,0,1))
    tau_fr = ca.cross(ca.vertcat(math.cos(fr_theta), math.sin(fr_theta), 0)*l, T[1]*ca.vertcat(0,0,1))
    tau_rl = ca.cross(ca.vertcat(math.cos(rl_theta), math.sin(rl_theta), 0)*l, T[2]*ca.vertcat(0,0,1))
    tau_rr = ca.cross(ca.vertcat(math.cos(rr_theta), math.sin(rr_theta), 0)*l, T[3]*ca.vertcat(0,0,1))

    # Compute yaw torque from each rotor's spin direction
    tau_yaw_fl = T[0] * ca.vertcat(0,0,1) * force_to_torque * 1
    tau_yaw_fr = T[1] * ca.vertcat(0,0,1) * force_to_torque * -1
    tau_yaw_rl = T[2] * ca.vertcat(0,0,1) * force_to_torque * -1
    tau_yaw_rr = T[3] * ca.vertcat(0,0,1) * force_to_torque * 1

    # Total torque
    tau_yaw = tau_yaw_fl + tau_yaw_fr + tau_yaw_rl + tau_yaw_rr
    tau = tau_fl + tau_fr + tau_rl + tau_rr + tau_yaw
    alpha = ca.inv(I) @ tau                       # Angular acceleration

    # State derivatives: dx/dt = [v, omega, acc, alpha]
    dx = ca.vertcat(v, omg, acc_w, alpha)

    # Linearize the system around (x0, u0)
    A = ca.Function('A', [ca.vertcat(x, th, v, omg), u], [ca.jacobian(dx, ca.vertcat(x, th, v, omg))])
    B = ca.Function('B', [ca.vertcat(x, th, v, omg), u], [ca.jacobian(dx, u)])
    Ak = A(x0, u0).full()
    Bk = B(x0, u0).full()

    # Discretize the system using Euler method
    Ad = np.eye(12) + dt * Ak
    Bd = dt * Bk

    # Define MPC problem: state dim = 12, control dim = 4, horizon = n_steps
    nx, nu, N = 12, 4, n_steps
    Q = np.eye(nx) * 0.2; Q[0:3, 0:3] = np.eye(3) * 1.0  # State cost, prioritize position
    R = np.eye(nu) * 0.01                                # Control effort penalty

    if optimizer == 'qpoases':
        opti = ca.Opti('conic')
    else:
        opti = ca.Opti()
    
    X = opti.variable(nx, N+1)      # Predicted states
    U = opti.variable(nu, N)        # Control inputs

    # Initial state constraint
    opti.subject_to(X[:,0] == x0)

    # Dynamics and input constraints
    for k in range(N):
        opti.subject_to(X[:,k+1] == Ad @ X[:,k] + Bd @ U[:,k] - ca.vertcat(0, 0, 0, 0, 0, 0, 0, 0, 9.8, 0, 0, 0) * dt)
        opti.subject_to(opti.bounded(0.01, U[:,k], 0.99))  # Input bounds [0.01, 0.99]

    # Objective function
    cost = 0
    for k in range(N):
        e = X[:,k] - x_target
        cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U[:,k].T, R, U[:,k]])
    # Optional terminal cost
    eN = X[:,N] - x_target
    cost += ca.mtimes([eN.T, Q, eN])

    opti.minimize(cost)

    # Use IPOPT to solve the optimization problem (can replace with qpoases if installed)
    opti.solver(optimizer)
    sol = opti.solve()

    u_opt = np.asarray(sol.value(U[:,0])).flatten()  # First control input
    x_pred = np.asarray(sol.value(X))                # Predicted trajectory

    return u_opt, x_pred


