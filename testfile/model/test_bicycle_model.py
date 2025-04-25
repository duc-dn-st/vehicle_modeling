import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


class BicycleModel:
    def __init__(self):
        self._register_parameter()
        self._register_state()
        self._register_input()
        self._register_frame()

    def _register_parameter(self):
        self.l_r = 0.5
        self.l_f = 0.5
        self.length_base = self.l_r + self.l_f
        self.width = 0.7
        self.max_steering_angle = np.pi / 4
        self.max_acceleration = 1.0

    def _register_state(self):
        self.x_f = 0.0
        self.y_f = 0.0
        self.theta = 0.0
        self.state = np.array([self.x_f, self.y_f, self.theta])

    def _register_input(self):
        self.v_f = 0.0
        self.delta = 0.0

    def _register_frame(self):
        self.frames = []
        self.fig = plt.figure(figsize=(9, 9))
        self.main_ax = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
        self.main_ax.set_aspect('equal')
        self.main_ax.tick_params(
            labelbottom=True, labelleft=True, labelright=False, labeltop=False)
        self.main_ax.tick_params(
            bottom=True, left=True, right=False, top=False)
        self.main_ax.set_xlim(auto=True)
        self.main_ax.set_ylim(auto=True)
        self.main_ax.set_xlabel('X [m]')
        self.main_ax.set_ylabel('Y [m]')
        self.main_ax.grid(True)

        self.steer_ax = plt.subplot2grid((3, 3), (2, 0))
        self.velocity_ax = plt.subplot2grid((3, 3), (2, 1))

    def update_state(self, input, dt):
        self.state = self.state.reshape(-1, 3)
        self.v_f = input[0]
        self.delta = input[1]
        self.x_f += self.v_f * np.cos(self.delta + self.theta) * dt
        self.y_f += self.v_f * np.sin(self.delta + self.theta) * dt
        self.theta += (self.v_f / self.length_base) * np.sin(self.delta) * dt
        self.state = np.vstack([self.state, [self.x_f, self.y_f, self.theta]])

        frame = self._append_robot_frame()
        steer, vel = self._append_control_frame()
        frame += steer + vel

        self.frames.append(frame)

    def _append_robot_frame(self):
        vw, vl = self.width, self.length_base
        yaw = self.theta
        vehicle_shape_x = [-0.5*vl, -0.5*vl, +0.5*vl, +0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, -0.5*vw, -0.5*vw]
        vehicle_x, vehicle_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [
                                   self.x_f, self.y_f])

        robot_frame = self.main_ax.plot(
            vehicle_x, vehicle_y, color='blue', linewidth=0.5, zorder=3)

        # wheels
        wheel_w = 0.05
        wheel_r = 0.125
        wheel_shape_x = [-wheel_r, +wheel_r, +wheel_r, -wheel_r, -wheel_r]
        wheel_shape_y = [-wheel_w, -wheel_w, +wheel_w, +wheel_w, -wheel_w]

        wheel_shape_rl_x, wheel_shape_rl_y = \
            self._affine_transform(
                wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, 0.5*vw])
        wheel_rl_x, wheel_rl_y = \
            self._affine_transform(wheel_shape_rl_x, wheel_shape_rl_y, yaw, [
                                   self.x_f, self.y_f])

        robot_frame += self.main_ax.fill(wheel_rl_x,
                                         wheel_rl_y, color='black', zorder=3)

        wheel_shape_rl_x, wheel_shape_rl_y = \
            self._affine_transform(
                wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, -0.5*vw])
        wheel_rl_x, wheel_rl_y = \
            self._affine_transform(wheel_shape_rl_x, wheel_shape_rl_y, yaw, [
                                   self.x_f, self.y_f])

        robot_frame += self.main_ax.fill(wheel_rl_x,
                                         wheel_rl_y, color='black', zorder=3)

        wheel_shape_f_x, wheel_shape_f_y = \
            self._affine_transform(
                wheel_shape_x, wheel_shape_y, self.delta, [0.4*vl, 0.0])
        wheel_f_x, wheel_f_y = \
            self._affine_transform(wheel_shape_f_x, wheel_shape_f_y, yaw, [
                                   self.x_f, self.y_f])

        robot_frame += self.main_ax.fill(wheel_f_x,
                                         wheel_f_y, color='black', zorder=3)

        robot_frame += self.main_ax.plot(
            self.state[:, 0], self.state[:, 1], color='red', linewidth=0.5, zorder=3)

        return robot_frame

    def _append_control_frame(self):
        # steering angle
        steer = np.abs(self.delta)

        if self.delta < 0.0:
            steer_pie_obj, _ = self.steer_ax.pie([self.max_steering_angle, steer, (self.max_steering_angle - steer),
                                                  2*self.max_steering_angle], startangle=180, counterclock=False,
                                                 colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4})

        else:
            steer_pie_obj, _ = self.steer_ax.pie([(self.max_steering_angle - steer), steer, self.max_steering_angle,
                                                  2*self.max_steering_angle], startangle=180, counterclock=False,
                                                 colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4})

        steer_frame = steer_pie_obj
        steer_frame += [self.steer_ax.text(0, -1, f"{np.rad2deg(-self.delta):+.2f} " + r"$ \rm{[deg]}$", size=14,
                                           horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # velocity
        velocity = np.abs(self.v_f)
        if velocity > 0.0:
            velocity_pie_obj, _ = self.velocity_ax.pie([self.max_acceleration, velocity, (self.max_acceleration - velocity),
                                                        2*self.max_acceleration], startangle=180, counterclock=False,
                                                       colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4})

        else:
            velocity_pie_obj, _ = self.velocity_ax.pie([(self.max_acceleration - velocity), velocity, self.max_acceleration,
                                                        2*self.max_acceleration], startangle=180, counterclock=False,
                                                       colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4})

        velocity_frame = velocity_pie_obj
        velocity_frame += [self.velocity_ax.text(0, -1, f"{self.v_f:+.2f} " + r"$ \rm{[deg]}$", size=14,
                                                 horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        return steer_frame, velocity_frame

    def _affine_transform(self, xlist, ylist, angle, translation=[0.0, 0.0]):
        transformed_x = []
        transformed_y = []
        for x, y in zip(xlist, ylist):
            transformed_x.append(x * np.cos(angle) - y *
                                 np.sin(angle) + translation[0])
            transformed_y.append(x * np.sin(angle) + y *
                                 np.cos(angle) + translation[1])

        transformed_x.append(transformed_x[0])
        transformed_y.append(transformed_y[0])

        return transformed_x, transformed_y

    def show_animation(self, interval_ms):
        self.ani = ArtistAnimation(self.fig, self.frames,
                                   interval=interval_ms, repeat=True)

        plt.show()


bicycle_model = BicycleModel()
total_time = 15
dt = 0.1
steps = int(total_time / dt)
interval = 5

for step in range(steps):
    velocity = np.random.uniform(0.2, 0.7)
    steering = (np.pi / 6) * np.sin(0.5 * step * dt)

    bicycle_model.update_state([velocity, steering], dt)

bicycle_model.show_animation(interval)
