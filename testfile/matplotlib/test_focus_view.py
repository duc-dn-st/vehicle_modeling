import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Setup
fig, ax = plt.subplots()
robot_dot, = ax.plot([], [], 'ro', markersize=8)  # red dot: robot
path_line, = ax.plot([], [], '0.7')  # gray line for reference path

view_range = 10  # camera size

# Create a circular path
t_values = np.linspace(0, 2 * np.pi, 500)
x_path = 20 * np.cos(t_values)
y_path = 20 * np.sin(t_values)

def init():
    robot_dot.set_data([], [])
    path_line.set_data(x_path, y_path)
    return robot_dot, path_line

def update(frame):
    x = x_path[frame % len(x_path)]
    y = y_path[frame % len(y_path)]

    # Update robot position
    robot_dot.set_data(x, y)

    # Keep camera centered on robot
    ax.set_xlim(x - view_range/2, x + view_range/2)
    ax.set_ylim(y - view_range/2, y + view_range/2)

    return robot_dot, path_line

ani = animation.FuncAnimation(
    fig, update, frames=len(x_path),
    init_func=init, blit=False, interval=30, repeat=True
)

plt.show()
