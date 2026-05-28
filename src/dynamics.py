from orbital_tools import Atmosphere
from orbital_tools import rk4_step
from orbital_tools import Dynamics
import numpy as np

class OrbitalDynamics:

    def __init__(self, height, dt):
        self.height = height
        self.dt = dt
        self.atm = Atmosphere()
        self.dyn = Dynamics()

    def orbit_propagation(self):
            
            height_km = self.height*1e3   
            r0_mag = self.atm.r_earth + height_km

            # Initial time parameters
            T = 2*np.pi*np.sqrt((r0_mag**3)/self.atm.mu_earth)

            # Initial Positional parameters
            r0 = np.array([r0_mag, 0.0, 0.0]) 
            vcirc = np.sqrt(self.atm.mu_earth/r0_mag)
            v0 = np.array([0.0, vcirc, 0.0])
            y0 = np.hstack((r0, v0))
            y = y0.copy()

            y = rk4_step(Dynamics.two_body_rhs, 0, y)

            return y
                