import numpy as np
from models.bicycle_model import BicycleModel
# from simulators.time_stepping import TimeStepping
from animation.animation import Animation


total_time = 15
dt = 0.1
steps = int(total_time / dt)
interval = 5

model = BicycleModel()

# simulator = TimeStepping(model)

animation = Animation(model)

for step in range(steps):
    velocity = np.random.uniform(0.2, 0.7)

    steering = (np.pi / 6) * np.sin(0.5 * step * dt)

    steering = 0.0

    model._update_state([velocity, steering], dt)

    animation._register_animation([velocity, steering], dt)

animation.show_animation(interval)
