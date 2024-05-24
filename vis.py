from utils import *
import config
from Drone import Drone
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create Drone instance
drone = Drone(init_pose=config.init_pose,
              target_pose=config.target_pose)

# Set thrust to perfectly counteract gravity
g_force = 9.8 * config.mass #N
g_force_per_motor = g_force/4
thrust_per_motor = g_force_per_motor/config.max_thrust
# drone.set_thrusts(thrust_per_motor-1, thrust_per_motor-1, thrust_per_motor-1, thrust_per_motor+1)
drone.set_thrusts(thrust_per_motor, thrust_per_motor, thrust_per_motor, thrust_per_motor)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.xlim(-10, 10)
plt.ylim(-10, 10)

ax.set_xlim(-10, 10)  # Set x-axis limits to a range around zero
ax.set_ylim(-10, 10)  # Set y-axis limits to a range around zero
ax.set_zlim(0, 25)

# Show plot and update without closing
plt.ion()  # Turn on interactive mode
plt.show()

while True:
    print(drone.thrust)
    print(f"Visualizing... | {drone.pose.x} | {drone.pose.theta}")
    ax.scatter(drone.pose.x.x, drone.pose.x.y, drone.pose.x.z, marker='o')

    dir_travel, dir_normal = drone.thrust_normal_vector()
    ax.quiver(drone.pose.x.x, drone.pose.x.y, drone.pose.x.z, dir_travel.x, dir_travel.y, dir_travel.z, pivot='tail', length=3, normalize=True)
    ax.quiver(drone.pose.x.x, drone.pose.x.y, drone.pose.x.z, dir_normal.x, dir_normal.y, dir_normal.z, pivot='tail', length=3, normalize=True)

    plt.pause(0.01)
    ax.cla() 
    ax.set_xlim(-10, 10)  # Set x-axis limits to a range around zero
    ax.set_ylim(-10, 10)  # Set y-axis limits to a range around zero
    ax.set_zlim(0, 25)
    

    dt = 0.01 #sec
    state, reward, done = drone.update(dt)
    print(f"     Reward: {reward} | Done: {done}")
    # time.sleep(3)
# Close the visualization window
vis.destroy_window()