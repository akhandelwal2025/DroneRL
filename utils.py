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
    
class Thrust:
    # fl, fr, rl, rr are represented as percents (i.e. front-left = 0.1 = 10% thrust)
    # allows policy to output logits that are directly used to control thrust
    # max thrust is in Newtons
    def __init__(self, fl=0, fr=0, rl=0, rr=0, max_thrust=100):
        self.max_thrust = max_thrust
        self.fl = Vector3(0, 0, fl * self.max_thrust)
        self.fr = Vector3(0, 0, fr * self.max_thrust)
        self.rl = Vector3(0, 0, rl * self.max_thrust)
        self.rr = Vector3(0, 0, rr * self.max_thrust)
    
    def set_thrusts(self, fl, fr, rl, rr):
        # fl, fr, rl, rr are in %
        self.fl = Vector3(0, 0, fl * self.max_thrust)
        self.fr = Vector3(0, 0, fr * self.max_thrust)
        self.rl = Vector3(0, 0, rl * self.max_thrust)
        self.rr = Vector3(0, 0, rr * self.max_thrust)
    
    def sum(self):
        return self.fl + self.fr + self.rl + self.rr