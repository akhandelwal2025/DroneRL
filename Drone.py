from utils import *
import config
import numpy as np

class Drone:
    def __init__(self, init_x, init_theta, init_v, init_omega, init_a, init_alpha):
        self.pose = Pose(init_x, init_theta, init_v, init_omega, init_a, init_alpha)
        self.thrust = Thrust(max_thrust=config.max_thrust)
    
    def calc_torques(self):
        # pitch, roll torques created by rotors away from COM
        tau_fl = Vector3.cross(config.fl_r, self.thrust.fl)
        tau_fr = Vector3.cross(config.fr_r, self.thrust.fr)
        tau_rl = Vector3.cross(config.rl_r, self.thrust.rl)
        tau_rr = Vector3.cross(config.rr_r, self.thrust.rr)

        # yaw torques created by mismatch
        tau_yaw = self.thrust.sum() * config.force_to_torque # MAYBE?
        
        # return sum of torques
        return tau_fl + tau_fr + tau_rl + tau_rr + tau_yaw
    
    def thrust_normal_vector(self):
        body_to_inertial = self.body_to_inertial()
        f_total_body = self.thrust.sum()
        return Vector3.from_numpy(body_to_inertial @ f_total_body.to_numpy())
    
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
        print(self.pose.a)
    
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



    