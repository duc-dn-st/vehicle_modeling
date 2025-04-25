import numpy as np
import sys
sys.path.append('.')

from models.bicycle_model import BicycleModel
from references.reference import Reference


class PID:
    def __init__(self, model, reference):
        """! Constructor
        """
        self.model = model

        self.pre_long_e = 0.0

        self.pre_integral_long_ie = 0.0

        self.pre_head_e = 0.0

        self.pre_integral_head_ie = 0.0

        self.previous_index = 0

        self._register_controller_parameters()

        self.register_reference(reference)

    def _register_controller_parameters(self):
        """! Register controller parameters
        """
        self.kp = +1.2

        self.ki = +0.4

        self.kd = +0.1

    def register_reference(self, reference):
        """! Register reference
        """
        self.target_state = reference

    def _calculate_tracking_error(self, observed_state):
        """! Calculate tracking error
        """
        idx = self._get_nearest_reference(observed_state)

        x1, y1 = self.target_state[idx, 0], self.target_state[idx, 1]

        x2, y2 = self.target_state[idx, 0] + 1.0 * np.cos(self.target_state[idx, 2]), self.target_state[idx, 1] + 1.0*np.sin(self.target_state[idx, 2])

        vx, vy = x2 - x1, y2 - y1

        wx, wy = observed_state[0] - x1,  observed_state[1] - y1

        s = vx * wy - vy * wx

        tracking_error = -np.sign(s) * np.sqrt((self.target_state[idx, 0] - observed_state[0])**2 + (
            self.target_state[idx, 1] - observed_state[1])**2)

        return tracking_error

    def _get_nearest_reference(self, observed_state):
        """! Get nearest reference
        """
        search_index_length = 10

        previous_index = self.previous_index

        dx = [observed_state[0] - self.target_state[i, 0]
              for i in range(previous_index, previous_index + search_index_length)]

        dy = [observed_state[1] - self.target_state[i, 1]
              for i in range(previous_index, previous_index + search_index_length)]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        min_d = np.min(d)

        nearest_index = self.previous_index + d.index(min_d)

        self.previous_index = nearest_index

        return nearest_index

    def calculate_input(self, observed_state, dt):
        """! Calculate control input
        """
        long_e = self._calculate_tracking_error(observed_state)

        long_ie = self.pre_integral_long_ie + \
            (long_e + self.pre_long_e) * dt / 2.0

        long_de = (long_e - self.pre_long_e) / dt

        steering = self.kp * long_e + self.ki * long_ie + self.kd * long_de

        if steering >= self.model.max_steering_angle:

            steering = self.model.max_steering_angle

        elif steering <= -self.model.max_steering_angle:

            steering = -self.model.max_steering_angle

        self.pre_long_e = long_e

        return steering