from utils import *
import math
import numpy as np

# -------- Drone Parameters --------
mass = 0.468 #kg
max_thrust = 10 #N
rotor_arm_length = 0.225 #m

lift_coeff = 2.980 * 10e-6 #k
drag_coeff = 1.140 * 10e-7 #b
force_to_torque = drag_coeff / lift_coeff #f = k*omega^2 | tau = b*omega^2

Ixx = 4.856 * 10e-3
Iyy = 4.856 * 10e-3
Izz = 8.801 * 10e-3

# rotor arms, theta from +x
fr_theta = math.radians(45)
fl_theta = math.radians(135)
rl_theta = math.radians(225)
rr_theta = math.radians(315)

# CW (+1), CCW (-1) config
"""
1   -1

-1   1
"""
fl_rot_dir = 1
fr_rot_dir = -1
rl_rot_dir = -1
rr_rot_dir = 1

fr_r = Vector3(math.cos(fr_theta), math.sin(fr_theta), 0) * rotor_arm_length
fl_r = Vector3(math.cos(fl_theta), math.sin(fl_theta), 0) * rotor_arm_length
rl_r = Vector3(math.cos(rl_theta), math.sin(rl_theta), 0) * rotor_arm_length
rr_r = Vector3(math.cos(rr_theta), math.sin(rr_theta), 0) * rotor_arm_length

init_pose = Pose(x=Vector3(2, 5, 1), 
                 theta=Vector3(0, 0, 0),
                 v=Vector3(0, 0, 0),
                 omega=Vector3(0, 0, 0),
                 a=Vector3(0, 0, 0),
                 alpha=Vector3(0, 0, 0))

target_pose = Pose(x=Vector3(0, 0, 10), 
                   theta=Vector3(0, 0, 0),
                   v=Vector3(0, 0, 0),
                   omega=Vector3(0, 0, 0),
                   a=Vector3(0, 0, 0),
                   alpha=Vector3(0, 0, 0))

# -------- PPO Parameters --------
STATE_DIM = 9
ACTION_DIM = 4

# Each batch contains multiple episodes
# An episode is a single trajectory rollout (starting from init state then following policy to terminal state)
NUM_BATCHES = 64
EPISODES_PER_BATCH = 128
PPO_EPOCHS_PER_BATCH = 5 # arbitrarily chosen

DISCOUNT_FACTOR = 0.99
EPSILON = 0.2 # arbitrarily chosen, recommended by original PPO paper?

INIT_LR = 0.005 # ADAMW Optimizer
