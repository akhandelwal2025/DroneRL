import numpy as np

class Vector3:
    def __init__(self, x, y, z):
        self.x = round(x, 5)
        self.y = round(y, 5)
        self.z = round(z, 5)

    def __add__(self, addend):
        if isinstance(addend, Vector3):
            return Vector3(self.x + addend.x, self.y + addend.y, self.z + addend.z)
        else:
            raise Exception(f"Vector3 only supports addition with (Vector3). Addend provided is of type {type(addend)}")

    def __sub__(self, subtrahend):
        if isinstance(subtrahend, Vector3):
            return Vector3(self.x - subtrahend.x, self.y - subtrahend.y, self.z - subtrahend.z)
        else:
            raise Exception(f"Vector3 only supports addition with (Vector3). Subtrahend provided is of type {type(subtrahend)}")

    def __mul__(self, multiplicand):
        if isinstance(multiplicand, (int, float)):
            return Vector3(self.x * multiplicand, self.y * multiplicand, self.z * multiplicand)
        else:
            raise Exception(f"Vector3 only supports multiplication with (int, float). Multiplicand provided is of type {type(multiplicand)}")

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            return Vector3(self.x / divisor, self.y / divisor, self.z / divisor)
        else:
            raise Exception(f"Vector3 only supports division with (int, float). Divisor provided is of type {type(divisor)}")

    def __str__(self):
        return f"X: {self.x} | Y: {self.y} | Z: {self.z}"

    def l2_norm(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def to_numpy(self):
        return np.asarray([self.x, self.y, self.z])
    
    def to_list(self):
        return [self.x, self.y, self.z]
    
    @staticmethod
    def from_numpy(np_arr):
        if np_arr.size == 3:
            return Vector3(np_arr[0], np_arr[1], np_arr[2])
        else:
            raise Exception(f"np_arr passed to from_numpy has {np_arr.size} elements. The expected number is 3")

    @staticmethod
    def cross(vec1, vec2):
        cx = vec1.y * vec2.z - vec1.z * vec2.y
        cy = vec1.z * vec2.x - vec1.x * vec2.z
        cz = vec1.x * vec2.y - vec1.y * vec2.x
        return Vector3(cx, cy, cz)
    
class Pose:
    def __init__(self, x: Vector3, theta: Vector3, v: Vector3, omega: Vector3, a: Vector3, alpha: Vector3):
        # theta.x = pitch
        # theta.y = roll
        # theta.z = yaw
        self.x = x
        self.theta = theta
        self.v = v
        self.omega = omega
        self.a = a
        self.alpha = alpha
    
    def to_numpy(self):
        return np.concatenate((self.x.to_numpy(),
                               self.theta.to_numpy(), 
                               self.v.to_numpy(),
                               self.omega.to_numpy(),
                               self.a.to_numpy(),
                               self.alpha.to_numpy()), axis=0)
    
class Thrust:
    # fl, fr, rl, rr are represented as percents (i.e. front-left = 0.1 = 10% thrust)
    # allows policy to output logits that are directly used to control thrust
    # max thrust is in Newtons
    def __init__(self, fl=0, fr=0, rl=0, rr=0, max_thrust=10):
        self.max_thrust = max_thrust
        self.fl = Vector3(0, 0, fl * self.max_thrust)
        self.fr = Vector3(0, 0, fr * self.max_thrust)
        self.rl = Vector3(0, 0, rl * self.max_thrust)
        self.rr = Vector3(0, 0, rr * self.max_thrust)
    
    def __str__(self):
        return f"fl = {self.fl.z}N | fr = {self.fr.z}N | rl = {self.rl.z}N | rr = {self.rr.z}N"
    
    def set_thrusts(self, fl, fr, rl, rr):
        # fl, fr, rl, rr are in %
        if fl < 0 or fr < 0 or rl < 0 or rr < 0:
            raise Exception(f"Thrusts must be >=0. requested_fl = {fl*100}% | requested_fr = {fr*100}% | requested_rl = {rl*100}% | requested_rr = {rr*100}%")
        
        if fl > 1 or fr > 1 or rl > 1 or rr > 1:
            raise Exception(f"Thrusts must be <=1. requested_fl = {fl*100}% | requested_fr = {fr*100}% | requested_rl = {rl*100}% | requested_rr = {rr*100}%")
        
        self.fl = Vector3(0, 0, fl * self.max_thrust)
        self.fr = Vector3(0, 0, fr * self.max_thrust)
        self.rl = Vector3(0, 0, rl * self.max_thrust)
        self.rr = Vector3(0, 0, rr * self.max_thrust)
    
    def sum(self):
        return self.fl + self.fr + self.rl + self.rr
    
class Episode:
    # an episode represents a single rollout
    # start in init state, then follow policy until terminal state
    # record all states, actions, rewards, log probs achieved until then
    def __init__(self, eps_num, discount_factor):
        self.eps_num = eps_num
        self.discount_factor = discount_factor
        self.states = [] # Pose
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.rewards_to_go = []
        self.log_probs = []

    def add_state(self, state):
        self.states.append(state)
    
    def add_action(self, action):
        self.actions.append(action)
    
    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def add_log_prob(self, log_prob):
        self.log_probs.append(log_prob)

    def get_states(self):
        # self.states.shape = Nx18, each pose contains 18 elements (x, v, a, theta, omega, alpha)
        # return it as Nx18 matrix
        N = len(self.states)
        to_ret = np.empty((N, 18))
        for i, state in enumerate(self.states):
            to_ret[i] = state.to_numpy()
        return to_ret
    
    def get_log_probs(self):
        # self.log_probs.shape = Nx4, N = num states, 4 = log_prob of each element in the action
        return np.asarray(self.log_probs)
    
    def get_advantages(self):
        return self.advantages

    def get_rewards_to_go(self):
        return self.rewards_to_go
    
    def compute_discounted_reward_to_go(self):
        running = 0
        for i, reward in enumerate(self.rewards):
            running += reward * (self.discount_factor ** i)
            self.rewards_to_go.append(running)
        self.rewards_to_go[::-1]
        self.rewards_to_go = np.asarray(self.rewards_to_go)
    
    def compute_advantages(self, values):
        # values = np.ndarray, Nx1 - N = num states
        # TODO: Check this logic
        # there are N states. in train.py, we don't add the terminal state. therefore, values.size == rewards_to_go.size
        assert(len(values) == len(self.rewards_to_go))
        for i in range(len(values)):
            self.advantages.append(self.rewards_to_go[i] - values[i])
        
        # implement advantage normalization for numerical stability if needed
        self.advantages = np.asarray(self.advantages)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-10) # 1e-10 added to prevent division by zero

class Batch:
    # a batch consists of multiple episodes
    def __init__(self, batch_num, eps_per_batch):
        self.batch_num = batch_num
        self.eps_per_batch = eps_per_batch
        self.episodes = []
        self.all_states = []
        self.all_log_probs = []
        self.all_advantages = []
        self.all_rewards_to_go = []

    def add_eps(self, eps):
        self.episodes.append(eps)

    def process_all_eps(self):
        for eps in self.episodes:
            self.all_states.append(eps.get_states())
            self.all_log_probs.append(eps.get_log_probs())
            self.all_advantages.append(eps.get_advantages())
            self.all_rewards_to_go.append(eps.get_rewards_to_go())
        
        self.all_states = np.concatenate(self.all_states, axis=0) # (eps_per_batch * N, 18)
        self.all_log_probs = np.concatenate(self.all_log_probs, axis=0) # (eps_per_batch * N, 4)
        self.all_advantages = np.asarray(self.all_advantages) # (N,)
        self.all_rewards_to_go = np.asarray(self.all_rewards_to_go) # (N,)