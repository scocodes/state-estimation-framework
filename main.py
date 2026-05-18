import numpy as np

from src.filters.basic_kalman_filter import BasicKalman
from src.models.constant_velocity import constant_velocity_model

dt = 1.0

F, H = constant_velocity_model(dt)

x0 = np.array([[0, 0, 1, 0.5]]).T
P0 = np.eye(4) * 100
Q = np.eye(4) * 0.01
R = np.eye(2) * 4

kf = BasicKalman(x0, P0, F, H, Q, R)

true_state = x0.copy()

for i in range(10):
    true_state = F @ true_state
    z = H@ true_state + np.random.normal(0, 2, size=(2, 1))

    kf.predict()
    estimate = kf.update(z)

    print("True:", true_state.ravel())
    print("Measured:", z.ravel())
    print("Estimated:", estimate.ravel())
    print()
    

