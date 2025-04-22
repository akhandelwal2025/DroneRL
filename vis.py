from utils import *
import config
from Drone import Drone
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpc import mpc_predict_future_n_steps
from trajectories import gen_straight_line, gen_spiral

# Create Drone instance
# waypoints = gen_straight_line()
waypoints = gen_spiral()
drone = Drone(init_pose=config.init_pose,
              target_pose=waypoints[0])

# Set thrust to perfectly counteract gravity
g_force = 9.8 * config.mass #N
g_force_per_motor = g_force/4
thrust_per_motor = g_force_per_motor/config.max_thrust
# drone.set_thrusts(thrust_per_motor-1, thrust_per_motor-1, thrust_per_motor-1, thrust_per_motor+1)
drone.set_thrusts(thrust_per_motor+0.1, thrust_per_motor+0.1, thrust_per_motor+0.1, thrust_per_motor+0.1)

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

dt = 0.1 #sec
future_steps = 10
i = 0
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
    if np.linalg.norm(drone.pose.x.to_numpy() - drone.target_pose.x.to_numpy()) < 0.1:
        print(f"Successfully Reached Target Waypoint {drone.target_pose.x.to_numpy()}")
        i += 1
        drone.target_pose.x = Vector3.from_numpy(waypoints[i])
    
    x0 = drone.pose.to_numpy()[0:12]
    u0 = drone.thrust.to_numpy() / config.max_thrust
    x_target = drone.target_pose.to_numpy()[0:12]
    u_opt, x_pred = mpc_predict_future_n_steps(x0, u0, x_target, future_steps, dt)
    drone.set_thrusts(u_opt[0], u_opt[1], u_opt[2], u_opt[3])
    # plot predicted trajectory
    # ax.plot(x_pred[0, :], x_pred[1, :], x_pred[2, :], marker='o', color='green')

    dt = 0.1 #sec
    state, reward, done = drone.update(dt)

    # plot completed waypoints in green
    completed_xs = [waypoint[0] for waypoint in waypoints[:i]]
    completed_ys = [waypoint[1] for waypoint in waypoints[:i]]
    completed_zs = [waypoint[2] for waypoint in waypoints[:i]]
    ax.scatter(completed_xs, completed_ys, completed_zs, marker='o', color='green')

    # plot to-go waypoints in red
    to_go_xs = [waypoint[0] for waypoint in waypoints[i:]]
    to_go_ys = [waypoint[1] for waypoint in waypoints[i:]]
    to_go_zs = [waypoint[2] for waypoint in waypoints[i:]]
    ax.scatter(to_go_xs, to_go_ys, to_go_zs, marker='o', color='red')

    # time.sleep(3)
# Close the visualization window
vis.destroy_window()