import numpy as np
import matplotlib.pyplot as plt

# generate waypoints along a defined trajectory by sampling different parametric functions
def gen_straight_line(start_pose = np.asarray([0., 0., 0.]), 
                      target_pose = np.asarray([10., 10., 10.]),
                      num_waypoints = 25):
    t = 0
    dt = 1/num_waypoints
    curr_pose = start_pose
    waypoints = [curr_pose]
    for i in range(num_waypoints):
        t += dt
        next_pose = curr_pose.copy() + (target_pose - start_pose) * dt
        waypoints.append(next_pose)
        curr_pose = next_pose   
    return waypoints

NUM_SPIRAL_PERIODS = 5
def gen_spiral(spiral_periods = 5,
               num_waypoints = 250):
    t = 0
    dt = (spiral_periods * 2 * np.pi) / num_waypoints
    curr_pose = np.asarray([0., 0., 0.])
    waypoints = [curr_pose]
    for i in range(num_waypoints):
        t += dt
        next_pose = np.asarray([np.sin(t),
                                np.cos(t),
                                t])
        waypoints.append(next_pose)
    return waypoints
 
waypoints = gen_straight_line()
# waypoints = gen_spiral()
print(waypoints)
xs = [waypoint[0] for waypoint in waypoints]
ys = [waypoint[1] for waypoint in waypoints]
zs = [waypoint[2] for waypoint in waypoints]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, marker='o', color='red')
plt.show()
