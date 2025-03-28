import sys
sys.path.append('.')
import numpy as np
import pandas as pd

from models.bicycle_model import BicycleModel
from animation.animation import Animation
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class PID:
    def   __init__(self, model, reference):
        """! Constructor
        """
        self.model = model

        self.pre_long_e = 0.0

        self.pre_integral_long_ie = 0.0

        self.pre_head_e = 0.0

        self.pre_integral_head_ie = 0.0

        self.previous_index = 0

        self._register_controller_parameters()

        self._register_reference(reference)

    def _register_controller_parameters(self):
        """! Register controller parameters
        """
        self.kp = +1.2

        self.ki = +0.4

        self.kd = +0.1

    def _register_reference(self, reference):
        """! Register reference
        """
        self.target_state = reference
    
    def _calculate_tracking_error(self, observed_state):
        """! Calculate tracking error
        """
        idx = self._get_nearest_reference(observed_state)

        longitudinal_error = np.sqrt((self.target_state[idx, 0] - observed_state[0])**2 + (self.target_state[idx, 1] - observed_state[1])**2)

        heading_error = self.target_state[idx, 2] - observed_state[2]

        return longitudinal_error, heading_error
    
    def _calculate_velocity(self, observed_state, dt):
        pass
    
    def _calculate_steering(self, observed_state, dt):
        pass

    def _get_nearest_reference(self, observed_state):
        """! Get nearest reference
        """
        search_index_length = 10

        previous_index = self.previous_index

        dx = [observed_state[0] - self.target_state[i, 0] for i in range(previous_index, previous_index + search_index_length)]

        dy = [observed_state[1] - self.target_state[i, 1] for i in range(previous_index, previous_index + search_index_length)]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        min_d = np.min(d)

        nearest_index = self.previous_index + d.index(min_d)

        self.previous_index = nearest_index

        return nearest_index
    
model = BicycleModel()

model.x_f, model.y_f, model.theta = 0.0, 0.2, 0.0

model.state = np.array([model.x_f, model.y_f, model.theta])

animation = Animation(model)

# reference trajectory
data = pd.read_csv('references/ovalpath.csv')

ref_x = data['x'].values

ref_y = data['y'].values

ref_theta = data['yaw'].values

reference = np.vstack([ref_x, ref_y, ref_theta]).T

# simulation settings
sim_step = len(ref_x) # [step]

delta_t = 0.1 # [s]

# controller settings
controller = PID(model, reference)

for i in range(sim_step):

    animation._register_animation([velocity, steering], delta_t)

animation.show_animation(0.1)