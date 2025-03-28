import numpy as np


class BicycleModel:
    def __init__(self):
        """! Constructor
        """
        self._register_parameter()

        self._register_state()

        self._register_input()

    def _register_parameter(self):
        """! Register parameters
        @param length_base: distance between front and rear wheel
        @param width: width of vehicle
        @param max_steering_angle: maximum steering angle
        @param max_acceleration: maximum acceleration
        """
        self.length_base = 1.0

        self.width = 0.7

        self.max_steering_angle = np.pi / 4

        self.max_acceleration = 1.0

    def _register_state(self):
        """! Register states
        @param x_f: x position of front wheel
        @param y_f: y position of front wheel
        @param theta: heading angle
        """
        self.x_f = 0.0

        self.y_f = 0.0

        self.theta = 0.0

        self.state = np.array([self.x_f, self.y_f, self.theta])

    def _register_input(self):
        """! Register inputs
        @param v_f: forward velocity
        @param delta: steering angle
        """
        self.v_f = 0.0

        self.delta = 0.0

    def _update_state(self, input, dt):
        """! Update state with kinematic
        """
        self.state = self.state.reshape(-1, 3)

        self.v_f = input[0]

        self.delta = input[1]

        self.x_f += self.v_f * np.cos(self.delta + self.theta) * dt

        self.y_f += self.v_f * np.sin(self.delta + self.theta) * dt

        self.theta += (self.v_f / self.length_base) * np.sin(self.delta) * dt

        self.state = np.vstack([self.state, [self.x_f, self.y_f, self.theta]])
