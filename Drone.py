import copy
from utils import *
import config
import numpy as np

class Drone:
    def __init__(self, init_pose, target_pose):
        self.init_pose = init_pose # used to reset env 
        self.pose = init_pose # curr pose
        self.target_pose = target_pose
        self.thrust = Thrust(max_thrust=config.max_thrust)
        self.done = False

    def update_done(self):
        # the only terminal state is if the drone has hit the ground
        # TODO: maybe want to consider bounds for environment in x, y as well as terminal state
        # if too high rotation or acceleration
        self.done = self.pose.x.z <= 0
    
    def reset(self):
        self.pose = copy.deepcopy(self.init_pose)
        self.done = False
        return self.get_pose(), self.done
    
    def get_pose(self):
        return copy.deepcopy(self.pose)
    
    def set_thrusts(self, fl, fr, rl, rr):
        self.thrust.set_thrusts(fl, fr, rl, rr)

    def calc_reward(self):
        # reward formulation: max(0, 2 - (||p_t - p_target|| + 0.1 * ||a_t| + 0.1 * ||v_t| + 0.1 * ||omega_t||))
        rew_x = (self.pose.x - self.target_pose.x).l2_norm()
        rew_accel = 0.1 * self.pose.a.l2_norm()
        rew_vel = 0.1 * self.pose.v.l2_norm()
        rew_omega = 0.1 * self.pose.omega.l2_norm()
        return max(0, 2 - rew_x + rew_accel + rew_vel + rew_omega)

    def calc_torques(self):
        # pitch, roll torques created by rotors away from COM
        tau_fl = Vector3.cross(config.fl_r, self.thrust.fl)
        tau_fr = Vector3.cross(config.fr_r, self.thrust.fr)
        tau_rl = Vector3.cross(config.rl_r, self.thrust.rl)
        tau_rr = Vector3.cross(config.rr_r, self.thrust.rr)

        # yaw torques created by mismatch
        tau_yaw_fl = self.thrust.fl * config.force_to_torque * config.fl_rot_dir 
        tau_yaw_fr = self.thrust.fr * config.force_to_torque * config.fr_rot_dir
        tau_yaw_rl = self.thrust.rl * config.force_to_torque * config.rl_rot_dir
        tau_yaw_rr = self.thrust.rr * config.force_to_torque * config.rr_rot_dir
        tau_yaw = tau_yaw_fl + tau_yaw_fr + tau_yaw_rl + tau_yaw_rr
        
        # return sum of torques
        return tau_fl + tau_fr + tau_rl + tau_rr + tau_yaw
    
    def thrust_normal_vector(self):
        body_to_inertial = self.body_to_inertial()
        dir_travel = Vector3.from_numpy(body_to_inertial @ Vector3(0, 0, 1).to_numpy()) 
        dir_normal = Vector3.from_numpy(body_to_inertial @ Vector3(1, 0, 0).to_numpy())
        return dir_travel, dir_normal
    
    def body_to_inertial(self):
        Cx, Cy, Cz = np.cos(self.pose.theta.x), np.cos(self.pose.theta.y), np.cos(self.pose.theta.z)
        Sx, Sy, Sz = np.sin(self.pose.theta.x), np.sin(self.pose.theta.y), np.sin(self.pose.theta.z)
        """
        [
            [Cz * Cy, Cz * Sy * Sx - Sz * Cx, Cz * Sy * Cx + Sz * Sx],
            [Sz * Cy, Sz * Sy * Sx + Cz * Cx, Sz * Sy * Cx - Cz * Sx],
            [-Sy, Cy * Sx, Cy * Cx]
        ]
        """
        return np.asarray([
            [Cz * Cy, Cz * Sy * Sx - Sz * Cx, Cz * Sy * Cx + Sz * Sx],
            [Sz * Cy, Sz * Sy * Sx + Cz * Cx, Sz * Sy * Cx - Cz * Sx],
            [-Sy, Cy * Sx, Cy * Cx]
        ])

    def update_alpha(self):
        # tau = I @ alpha | ex. tau_x = Ixx * alpha_x b/c I = diagonal matrix with Ixx, Iyy, Izz on diagonal
        tau_total = self.calc_torques()
        self.pose.alpha = Vector3(tau_total.x / config.Ixx, tau_total.y / config.Iyy, tau_total.z / config.Izz)
    
    def update_omega(self, dt):
        delta_omega = self.pose.alpha * dt
        self.pose.omega += delta_omega
    
    def update_theta(self, dt):
        delta_theta = self.pose.omega * dt + self.pose.alpha * 0.5 * (dt ** 2) #theta = omega * dt + 1/2 * alpha * dt^2
        self.pose.theta += delta_theta
    
    def update_a(self):
        f_total_body = self.thrust.sum() # this is in body frame, always going to be +z. Need to convert this to inertial through euler rotation
        f_inertial = Vector3.from_numpy(self.body_to_inertial() @ f_total_body.to_numpy()) # this is now in inertial frame
        self.pose.a = f_inertial / config.mass
        self.pose.a.z -= 9.8 # m/s^2, subtract gravitational acceleration in z-dir
    
    def update_v(self, dt):
        delta_v = self.pose.a * dt
        self.pose.v += delta_v
    
    def update_x(self, dt):
        delta_x = self.pose.v * dt + self.pose.a * 0.5 * (dt ** 2) #x = v * dt + 1/2 * a * dt^2
        self.pose.x += delta_x
    
    def update(self, dt):
        # update rotational elements of pose
        self.update_alpha()
        self.update_omega(dt)
        self.update_theta(dt)

        # update translational elements of pose
        self.update_a()
        self.update_v(dt)
        self.update_x(dt)

        # update if in terminal state
        self.update_done()

        return self.get_pose(), self.calc_reward(), self.done


    