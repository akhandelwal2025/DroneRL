from utils import *
import config
from Drone import Drone
import time
import matplotlib.pyplot as plt

# Create Drone instance
drone = Drone(init_x=Vector3(0, 0, 10), 
              init_theta=Vector3(0, 0, 0),
              init_v=Vector3(0, 0, 0),
              init_omega=Vector3(0, 0, 0),
              init_a=Vector3(0, 0, 0),
              init_alpha=Vector3(0, 0, 0)) # z=+10

# Set thrust to perfectly counteract gravity
g_force = 9.8 * config.mass #N
g_force_per_motor = g_force/4
thrust_per_motor = g_force_per_motor/config.max_thrust + 0.05
drone.thrust.set_thrusts(thrust_per_motor, thrust_per_motor, thrust_per_motor, thrust_per_motor)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.xlim(-10, 10)
plt.ylim(-10, 10)

ax.set_xlim(-1e-13, 1e-13)  # Set x-axis limits to a range around zero
ax.set_ylim(-1e-14, 1e-14)  # Set y-axis limits to a range around zero
ax.set_zlim(0, 25)

# Show plot and update without closing
plt.ion()  # Turn on interactive mode
plt.show()

while True:
    print(f"Visualizing... | {drone.pose.x}")
    vis_x = drone.pose.x.x
    vis_y = drone.pose.x.y
    vis_z = drone.pose.x.z
    if vis_x < 10e-11:
        vis_x = 0
    if vis_y < 10e-11:
        vis_y = 0
    if vis_z < 10e-11:
        vis_z = 0
    ax.scatter(vis_x, vis_y, vis_z, marker='o')

    plt.pause(0.05)
    ax.cla() 
    ax.set_xlim(-1e-13, 1e-13)  # Set x-axis limits to a range around zero
    ax.set_ylim(-1e-14, 1e-14)  # Set y-axis limits to a range around zero
    ax.set_zlim(0, 25)
    

    dt = 0.1 #sec
    drone.update(dt)

# Close the visualization window
vis.destroy_window()