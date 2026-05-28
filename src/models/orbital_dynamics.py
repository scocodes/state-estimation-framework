from orbital_tools import Atmosphere
from orbital_tools import rk4_step
from orbital_tools import Dynamics
import numpy as np

class OrbitalDynamics:

    def __init__(self, dt, eps, height=1000):
        self.dt = dt
        self.atm = Atmosphere()
        self.dyn = Dynamics(self.atm)
        self.eps = eps
        self.height = height
    
    def initial_state(self, height):
        height_km = height*1e3   
        r0_mag = self.atm.r_earth + height_km

        # Initial Positional parameters
        r0 = np.array([r0_mag, 0.0, 0.0]) 
        vcirc = np.sqrt(self.atm.mu_earth/r0_mag)
        v0 = np.array([0.0, vcirc, 0.0])
        y0 = np.hstack((r0, v0))
        return y0

    def orbit_propagation(self):
        x0 = rk4_step(self.dyn.two_body_rhs, 0, self.initial_state(self.height), self.dt)
        return x0

    def finite_difference_jacobian(self):
        y0 = self.initial_state(self.height)
        n = len(y0)
        F = np.zeros((n,n))

        for i in range(n):
            x_plus = y0.copy()
            x_minus = y0.copy()

            x_plus[i] += self.eps
            x_minus[i] -= self.eps

            f_plus = rk4_step(self.dyn.two_body_rhs, 0, x_plus, self.dt)
            f_minus = rk4_step(self.dyn.two_body_rhs, 0, x_minus, self.dt)

            F[:, i] = (f_plus - f_minus)/ (2*self.eps)

        return F
        
         
