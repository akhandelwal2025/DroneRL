from utils import *
import math
import numpy as np

# Drone Parameters
mass = 0.468 #kg
max_thrust = 100 #N
rotor_arm_length = 0.2 #m

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

fr_r = Vector3(math.cos(fr_theta), math.sin(fr_theta), 0) * rotor_arm_length
fl_r = Vector3(math.cos(fl_theta), math.sin(fl_theta), 0) * rotor_arm_length
rl_r = Vector3(math.cos(rl_theta), math.sin(rl_theta), 0) * rotor_arm_length
rr_r = Vector3(math.cos(rr_theta), math.sin(rr_theta), 0) * rotor_arm_length


