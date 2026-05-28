from src.models.orbital_dynamics import OrbitalDynamics
import numpy as np
class Ekf:
    def __init__(self):
        self.orb_dyn = OrbitalDynamics(10, 1e-3)
    
    def covariance(self, F, P, Q):
        P_pred = F@P@F.T + Q
        return P_pred

    def extended_kalman_filter(self):
        x_pred = self.orb_dyn.orbit_propagation()
        F = self.orb_dyn.finite_difference_jacobian()

        R = np.eye(3) * 1e4
        Q = np.eye(6) * 1e-6
        P = np.eye(6) * 1e2

        p_pred = self.covariance(F, P, Q)

        H = np.array([[1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0]])

        z = x_pred[:3] + np.random.normal(0, 100, size=3)

        z_pred = H@x_pred

        y = z - z_pred

        S = H@p_pred@H.T + R
        K = p_pred @ H.T @ np.linalg.inv(S)

        x_update = x_pred + K@y

        I = np.eye(6)
        p_update = (I-K@H)@p_pred@(I-K@H).T + K@R@K.T

        print(x_pred, x_update)

        return x_update, p_update





