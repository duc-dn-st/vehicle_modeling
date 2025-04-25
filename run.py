import numpy as np
from models.bicycle_model import BicycleModel
from animation.animation import Animation
from references.reference import Reference
from controller.pid.pid import PID

model = BicycleModel()
model.x_f, model.y_f, model.theta = -0.1, 0.4, 0.0
model.state = np.array([model.x_f, model.y_f, model.theta])

reference = Reference()
reference = reference.register_reference('references/ovalpath')

controller = PID(model, reference)

animation = Animation(model, reference, controller)

# ===================================================================
total_time = 120
dt = 0.1
steps = int(total_time / dt)
interval = 5

for step in range(steps):
    velocity = np.random.uniform(0.6, 0.7)

    steering = controller.calculate_input(
        [model.x_f, model.y_f, model.theta], dt)

    model.update_state([velocity, steering], dt)

    model.v_f, model.delta = velocity, steering

    animation.update([model.x_f, model.y_f, model.theta])

    animation._register_animation([velocity, steering], dt)

animation.show_animation(interval)
