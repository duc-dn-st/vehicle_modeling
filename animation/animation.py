# External library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


class Animation:
    def __init__(self, model, reference, controller):
        """! Constructor
        """
        self.model = model

        self.reference = reference

        self.controller = controller

        self._register_frame()

    def _register_model(self, model):
        """! Register model
        """
        self.model = model

    def _register_animation(self, inputs, dt):
        """! Register animation"
        """
        frame = self._append_robot_frame()

        steer, vel = self._append_control_frame()

        frame += steer + vel

        self.frames.append(frame)

    def _register_frame(self):

        self.view_x_lim_min, self.view_x_lim_max = -7.5, 7.5

        self.view_y_lim_min, self.view_y_lim_max = -7.5, 7.5

        self.frames = []

        self.past_errors = []

        self.past_positions = []

        self.fig = plt.figure(figsize=(9, 9))

        self.main_ax = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)

        self.main_ax.set_aspect('equal')

        self.main_ax.set_xlim(self.view_x_lim_min, self.view_x_lim_max)

        self.main_ax.set_ylim(self.view_y_lim_min, self.view_y_lim_max)

        self.main_ax.tick_params(
            labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        self.main_ax.tick_params(
            bottom=False, left=False, right=False, top=False)

        self.main_ax.set_xlabel('X [m]')

        self.main_ax.set_ylabel('Y [m]')

        self.main_ax.grid(False)

        self.minimap_ax = plt.subplot2grid((3, 3), (2, 2))

        self.steer_ax = plt.subplot2grid((3, 3), (2, 0))

        self.steer_ax.set_title("Steering Angle", fontsize="12")

        self.velocity_ax = plt.subplot2grid((3, 3), (2, 1))

        self.velocity_ax.set_title("Velocity", fontsize="12")

        self.fig.tight_layout()

    def _append_robot_frame(self):

        vw, vl = self.model.width, self.model.length_base

        yaw = self.model.theta

        vehicle_shape_x = [-0.5*vl, -0.5*vl, +0.5*vl, +0.5*vl, -0.5*vl]

        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, -0.5*vw, -0.5*vw]

        vehicle_x, vehicle_y = \
            self._affine_transform(
                vehicle_shape_x, vehicle_shape_y, yaw, [0, 0])

        robot_frame = self.main_ax.plot(
            vehicle_x, vehicle_y, color='blue', linewidth=0.5, zorder=3)

        # wheels
        wheel_w = 0.05

        wheel_r = 0.125

        wheel_shape_x = [-wheel_r, +wheel_r, +wheel_r, -wheel_r, -wheel_r]

        wheel_shape_y = [-wheel_w, -wheel_w, +wheel_w, +wheel_w, -wheel_w]

        wheel_shape_rl_x, wheel_shape_rl_y = self._affine_transform(
            wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, 0.5*vw])

        wheel_rl_x, wheel_rl_y = self._affine_transform(
            wheel_shape_rl_x, wheel_shape_rl_y, yaw, [0.0, 0.0])

        robot_frame += self.main_ax.fill(
            wheel_rl_x, wheel_rl_y, color='black', zorder=3)

        wheel_shape_rl_x, wheel_shape_rl_y = self._affine_transform(
            wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, -0.5*vw])

        wheel_rl_x, wheel_rl_y = self._affine_transform(
            wheel_shape_rl_x, wheel_shape_rl_y, yaw, [0.0, 0.0])

        robot_frame += self.main_ax.fill(
            wheel_rl_x, wheel_rl_y, color='black', zorder=3)

        wheel_shape_f_x, wheel_shape_f_y = self._affine_transform(
            wheel_shape_x, wheel_shape_y, self.model.delta, [0.4*vl, 0.0])

        wheel_f_x, wheel_f_y = self._affine_transform(
            wheel_shape_f_x, wheel_shape_f_y, yaw, [0.0, 0.0])

        robot_frame += self.main_ax.fill(
            wheel_f_x, wheel_f_y, color='red', zorder=3)

        robot_frame += self.main_ax.plot(
            self.model.state[0] - self.model.x_f, self.model.state[1] - self.model.y_f, color='red', linewidth=0.5, zorder=3)

        ref_x = self.reference[:, 0] - \
            np.full(self.reference.shape[0], self.model.x_f)

        ref_y = self.reference[:, 1] - \
            np.full(self.reference.shape[0], self.model.y_f)

        robot_frame += self.main_ax.plot(ref_x,
                                         ref_y, color='black', linestyle='dashed')

        if len(self.past_positions) > 1:
            past_positions = np.array(self.past_positions)
            # Plot past positions in the global coordinate system
            robot_frame += self.main_ax.plot(
                past_positions[:, 0] - self.model.x_f + 0.4*vl*np.cos(yaw), past_positions[:, 1] - self.model.y_f + 0.4*vl*np.sin(yaw), 'r-', label='Robot', zorder=3
            )

        return robot_frame

    def _append_control_frame(self):
        steer = np.abs(self.model.delta)

        if self.model.delta < 0.0:
            steer_pie_obj, _ = self.steer_ax.pie(
                [self.model.max_steering_angle, steer,
                    (self.model.max_steering_angle - steer), 2*self.model.max_steering_angle],
                startangle=180, counterclock=False,
                colors=["lightgray", "black", "lightgray", "white"],
                wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4}
            )

        else:
            steer_pie_obj, _ = self.steer_ax.pie(
                [(self.model.max_steering_angle - steer), steer,
                    self.model.max_steering_angle, 2*self.model.max_steering_angle],
                startangle=180, counterclock=False,
                colors=["lightgray", "black", "lightgray", "white"],
                wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4}
            )

        steer_frame = steer_pie_obj

        steer_frame += [self.steer_ax.text(0, -1, f"{np.rad2deg(-self.model.delta):+.2f} " + r"$ \rm{[deg]}$", size=14,
                                           horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # velocity
        velocity = np.abs(self.model.v_f)

        if velocity > 0.0:
            velocity_pie_obj, _ = self.velocity_ax.pie(
                [self.model.max_acceleration, velocity,
                    (self.model.max_acceleration - velocity), 2*self.model.max_acceleration],
                startangle=180, counterclock=False,
                colors=["lightgray", "black", "lightgray", "white"],
                wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4}
            )

        else:
            velocity_pie_obj, _ = self.velocity_ax.pie(
                [(self.model.max_acceleration - velocity), velocity,
                    self.model.max_acceleration, 2*self.model.max_acceleration],
                startangle=180, counterclock=False,
                colors=["lightgray", "black", "lightgray", "white"],
                wedgeprops={'linewidth': 0, "edgecolor": "white", "width": 0.4}
            )

        velocity_frame = velocity_pie_obj

        velocity_frame += [self.velocity_ax.text(0, -1, f"{self.model.v_f:+.2f} " + r"$ \rm{[m/s]}$", size=14,
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

    def update(self, state):
        """! Update animation
        """
        self.model.state = state

        self.past_positions.append(state[:2])

    def show_animation(self, interval_ms):

        ani = ArtistAnimation(self.fig, self.frames,
                              interval=interval_ms, repeat=True)

        plt.show()
