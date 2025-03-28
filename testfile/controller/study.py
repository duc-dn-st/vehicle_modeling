import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from matplotlib import patches
from matplotlib.animation import ArtistAnimation



class Vehicle():
    def __init__(
            self,
            l_f: float = 1.5, # [m]
            l_r: float = 1.0, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[-30.0, 0.0], [30.0, 0.0]]),
            delta_t: float = 0.05, # [s]
            visualize: bool = True,
        ) -> None:
        """initialize vehicle environment
        state variables:
            x: x-axis position in the global frame [m]
            y: y-axis position in the global frame [m]
            yaw: orientation in the global frame [rad]
            v: longitudinal velocity [m/s]
        control input:
            steer: front tire angle of the vehicle [rad] (positive in the counterclockwize direction)
            accel: longitudinal acceleration of the vehicle [m/s^2] (positive in the forward direction)
        Note: dynamics of the vehicle is the Kinematic Bicycle Model. 
        """
        # vehicle parameters
        self.l_f = l_f # [m]
        self.l_r = l_r # [m]
        self.wheel_base = l_f + l_r # [m]
        self.max_steer_abs = max_steer_abs # [rad]
        self.max_accel_abs = max_accel_abs # [m/s^2]
        self.delta_t = delta_t #[s]
        self.ref_path = ref_path

        # visualization settings
        self.vehicle_w = 3.00
        self.vehicle_l = 4.00
        self.view_x_lim_min, self.view_x_lim_max = -20.0, 20.0
        self.view_y_lim_min, self.view_y_lim_max = -25.0, 25.0

        # reset environment
        self.visualize_flag = visualize
        self.reset()

    def reset(
            self, 
            init_state: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0]), # [x, y, yaw, v]
        ) -> None:
        """reset environment to initial state"""

        # reset state variables
        self.state = init_state

        # clear animation frames
        self.frames = []

        if self.visualize_flag:
            # prepare figure
            self.fig = plt.figure(figsize=(9,9))
            self.main_ax = plt.subplot2grid((3,4), (0,0), rowspan=3, colspan=3)
            self.minimap_ax = plt.subplot2grid((3,4), (0,3))
            self.steer_ax = plt.subplot2grid((3,4), (1,3))
            self.accel_ax = plt.subplot2grid((3,4), (2,3))

            # graph layout settings
            ## main view
            self.main_ax.set_aspect('equal')
            self.main_ax.set_xlim(self.view_x_lim_min, self.view_x_lim_max)
            self.main_ax.set_ylim(self.view_y_lim_min, self.view_y_lim_max)
            self.main_ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            self.main_ax.tick_params(bottom=False, left=False, right=False, top=False)
            ## mini map
            self.minimap_ax.set_aspect('equal')
            self.minimap_ax.axis('off')
            ## steering angle view
            self.steer_ax.set_title("Steering Angle", fontsize="12")
            self.steer_ax.axis('off')
            ## acceleration view
            self.accel_ax.set_title("Acceleration", fontsize="12")
            self.accel_ax.axis('off')
            
            # apply tight layout
            self.fig.tight_layout()

    def update(
            self, 
            u: np.ndarray, 
            delta_t: float = 0.0, 
            append_frame: bool = True, 
            vehicle_traj: np.ndarray = np.empty(0), # vehicle trajectory
        ) -> None:
        """update state variables"""
        # keep previous states
        x, y, yaw, v = self.state

        # prepare params
        l = self.wheel_base
        l_r = self.l_r
        dt = self.delta_t if delta_t == 0.0 else delta_t

        # limit control inputs
        steer = np.clip(u[0], -self.max_steer_abs, self.max_steer_abs)
        accel = np.clip(u[1], -self.max_accel_abs, self.max_accel_abs)

        """< CORE OF VEHICLE DYNAMICS >"""
        # update state variables
        beta = np.arctan(l_r / l * np.tan(steer))
        new_x = x + v * np.cos(yaw + beta) * dt
        new_y = y + v * np.sin(yaw + beta) * dt
        new_yaw = yaw + v / l * np.sin(beta) * dt
        new_v = v + accel * dt
        self.state = np.array([new_x, new_y, new_yaw, new_v])
        """< CORE OF VEHICLE DYNAMICS >"""

        # record frame
        if append_frame:
            self.append_frame(steer, accel, vehicle_traj)

    def get_state(self) -> np.ndarray:
        """return state variables"""
        return self.state.copy()

    def append_frame(self, steer: float, accel: float, vehicle_traj: np.ndarray) -> list:
        """draw a frame of the animation."""
        # get current states
        x, y, yaw, v = self.state

        ### main view ###
        # draw the vehicle shape
        vw, vl = self.vehicle_w, self.vehicle_l
        vehicle_shape_x = [-0.5*vl, -0.5*vl, +0.5*vl, +0.5*vl, -0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, -0.5*vw, -0.5*vw, 0.0]
        rotated_vehicle_shape_x, rotated_vehicle_shape_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [0, 0]) # make the vehicle be at the center of the figure
        frame = self.main_ax.plot(rotated_vehicle_shape_x, rotated_vehicle_shape_y, color='black', linewidth=2.0, zorder=3)

        # draw wheels
        ww, wl = 0.4, 0.7 #[m]
        wheel_shape_x = np.array([-0.5*wl, -0.5*wl, +0.5*wl, +0.5*wl, -0.5*wl, -0.5*wl])
        wheel_shape_y = np.array([0.0, +0.5*ww, +0.5*ww, -0.5*ww, -0.5*ww, 0.0])

        ## rear-left wheel
        wheel_shape_rl_x, wheel_shape_rl_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, 0.3*vw])
        wheel_rl_x, wheel_rl_y = \
            self._affine_transform(wheel_shape_rl_x, wheel_shape_rl_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_rl_x, wheel_rl_y, color='black', zorder=3)

        ## rear-right wheel
        wheel_shape_rr_x, wheel_shape_rr_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, -0.3*vw])
        wheel_rr_x, wheel_rr_y = \
            self._affine_transform(wheel_shape_rr_x, wheel_shape_rr_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_rr_x, wheel_rr_y, color='black', zorder=3)

        ## front-left wheel
        wheel_shape_fl_x, wheel_shape_fl_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, steer, [0.3*vl, 0.3*vw])
        wheel_fl_x, wheel_fl_y = \
            self._affine_transform(wheel_shape_fl_x, wheel_shape_fl_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_fl_x, wheel_fl_y, color='black', zorder=3)

        ## front-right wheel
        wheel_shape_fr_x, wheel_shape_fr_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, steer, [0.3*vl, -0.3*vw])
        wheel_fr_x, wheel_fr_y = \
            self._affine_transform(wheel_shape_fr_x, wheel_shape_fr_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_fr_x, wheel_fr_y, color='black', zorder=3)

        # draw the vehicle center circle
        vehicle_center = patches.Circle([0, 0], radius=vw/20.0, fc='white', ec='black', linewidth=2.0, zorder=4)
        frame += [self.main_ax.add_artist(vehicle_center)]

        # draw the reference path
        ref_path_x = self.ref_path[:, 0] - np.full(self.ref_path.shape[0], x)
        ref_path_y = self.ref_path[:, 1] - np.full(self.ref_path.shape[0], y)
        frame += self.main_ax.plot(ref_path_x, ref_path_y, color='black', linestyle="dashed", linewidth=1.5)

        # # draw the information text
        # text = "vehicle velocity = {v:>+6.1f} [m/s]".format(pos_e=x, head_e=np.rad2deg(yaw), v=v)
        # frame += [self.main_ax.text(0.5, 0.02, text, ha='center', transform=self.main_ax.transAxes, fontsize=14, fontfamily='monospace')]

        # # draw vehicle trajectory
        # if vehicle_traj.any():
        #     vehicle_traj_x_offset = np.append(np.ravel(vehicle_traj[:, 0]) - np.full(vehicle_traj.shape[0], x), [0.0])
        #     vehicle_traj_y_offset = np.append(np.ravel(vehicle_traj[:, 1]) - np.full(vehicle_traj.shape[0], y), [0.0])
        #     frame += self.main_ax.plot(vehicle_traj_x_offset, vehicle_traj_y_offset, color='purple', linestyle="solid", linewidth=2.0)

        # ### mini map view ###
        # frame += self.minimap_ax.plot(self.ref_path[:, 0], self.ref_path[:,1], color='black', linestyle='dashed')
        # rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap = \
        #     self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [x, y]) # make the vehicle be at the center of the figure
        # frame += self.minimap_ax.plot(rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap, color='black', linewidth=2.0, zorder=3)
        # frame += self.minimap_ax.fill(rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap, color='white', zorder=2)
        # if vehicle_traj.any():
        #     frame += self.minimap_ax.plot(vehicle_traj[:, 0], vehicle_traj[:, 1], color='purple', linestyle="solid", linewidth=1.0)

        ### control input view ###
        # steering angle
        MAX_STEER = self.max_steer_abs
        PIE_RATE = 3.0/4.0
        PIE_STARTANGLE = 225 # [deg]
        s_abs = np.abs(steer)
        if steer < 0.0: # when turning right
            steer_pie_obj, _ = self.steer_ax.pie([MAX_STEER*PIE_RATE, s_abs*PIE_RATE, (MAX_STEER-s_abs)*PIE_RATE, 2*MAX_STEER*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        else: # when turning left
            steer_pie_obj, _ = self.steer_ax.pie([(MAX_STEER-s_abs)*PIE_RATE, s_abs*PIE_RATE, MAX_STEER*PIE_RATE, 2*MAX_STEER*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        frame += steer_pie_obj
        frame += [self.steer_ax.text(0, -1, f"{np.rad2deg(steer):+.2f} " + r"$ \rm{[deg]}$", size = 14, horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # acceleration
        MAX_ACCEL = self.max_accel_abs
        PIE_RATE = 3.0/4.0
        PIE_STARTANGLE = 225 # [deg]
        a_abs = np.abs(accel)
        if accel > 0.0:
            accel_pie_obj, _ = self.accel_ax.pie([MAX_ACCEL*PIE_RATE, a_abs*PIE_RATE, (MAX_ACCEL-a_abs)*PIE_RATE, 2*MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        else:
            accel_pie_obj, _ = self.accel_ax.pie([(MAX_ACCEL-a_abs)*PIE_RATE, a_abs*PIE_RATE, MAX_ACCEL*PIE_RATE, 2*MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        frame += accel_pie_obj
        frame += [self.accel_ax.text(0, -1, f"{accel:+.2f} " + r"$ \rm{[m/s^2]}$", size = 14, horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # append frame
        self.frames.append(frame)

    # rotate shape and return location on the x-y plane.
    def _affine_transform(self, xlist: list, ylist: list, angle: float, translation: list=[0.0, 0.0]) -> Tuple[list, list]:
        transformed_x = []
        transformed_y = []
        if len(xlist) != len(ylist):
            print("[ERROR] xlist and ylist must have the same size.")
            raise AttributeError

        for i, xval in enumerate(xlist):
            transformed_x.append((xlist[i])*np.cos(angle)-(ylist[i])*np.sin(angle)+translation[0])
            transformed_y.append((xlist[i])*np.sin(angle)+(ylist[i])*np.cos(angle)+translation[1])
        transformed_x.append(transformed_x[0])
        transformed_y.append(transformed_y[0])
        return transformed_x, transformed_y

    def show_animation(self, interval_ms: int) -> None:
        """show animation of the recorded frames"""
        ani = ArtistAnimation(self.fig, self.frames, interval=interval_ms) # blit=True

        plt.show()

class PIDLongitudinalController():
    def __init__(
            self,
            p_gain: float = +1.2, # P Gain
            i_gain: float = +0.4, # I Gain
            d_gain: float = +0.1, # D Gain
            target_velocity: float = 3.0, # [m/s]
    ) -> None:
        """initialize pid controller for keeping constant velocity"""
        # pid control parameters
        self.K_p = p_gain
        self.K_i = i_gain
        self.K_d = d_gain
        self.target_vel = target_velocity
        self.pre_e = 0.0 # previous tracking error
        self.integrated_e = 0.0 # integrated tracking error

    def calc_control_input(self, observed_vel: float, delta_t: float) -> None:
        """calculate control input"""

        # calculate tracking error, its integral and derivative
        r = self.target_vel
        y = observed_vel
        e = r - y # tracking error to the traget velocity
        ie = self.integrated_e + (e + self.pre_e) * delta_t / 2.0 # integral of the tracking error
        de = (e - self.pre_e) / delta_t # derivative of the tracking error

        # calculate control input
        acc_cmd = self.K_p * e + self.K_i * ie + self.K_d * de

        # update previous tracking error
        self.pre_e = e

        return acc_cmd
    


# simulation settings
sim_step = 100 # [step]
delta_t = 0.1 # [s]

# initialize vehicle simulator
ref_path_x = np.linspace(-3.0, 50.0, 10)
ref_path_y = np.zeros(10)
vehicle = Vehicle(ref_path = np.array([ref_path_x, ref_path_y]).T, delta_t=delta_t)
vehicle.reset(init_state=np.array([0.0, 0.0, 0.0, 0.0])) # set initial state of the vehicle, [x, y, yaw, v]
vehicle_trajectory = np.array([vehicle.get_state()[:2]])

# initialize speed controller
speed_controller = PIDLongitudinalController(
    p_gain = 1.2,
    i_gain = 0.4,
    d_gain = 0.1,
    target_velocity = 3.0, # [m/s]
)

# simulation loop
for i in range(sim_step):

    # get current state of vehicle
    current_state = vehicle.get_state()
    current_velocity = current_state[3]

    # calculate control inputs
    steer_input = 0.0  # steering command [rad] # set zero because this is just a test run of speed controller
    accel_input = speed_controller.calc_control_input(observed_vel=current_velocity, delta_t=delta_t) # acceleration command [m/s^2]
    vehicle.update(u=[steer_input, accel_input], delta_t=delta_t, vehicle_traj=vehicle_trajectory) # update vehicle state
    vehicle_trajectory = np.vstack((vehicle_trajectory, vehicle.get_state()[:2])) # record vehicle trajectory

# show animation on jupyter notebook
vehicle.show_animation(interval_ms=delta_t*1000)