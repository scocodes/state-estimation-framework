import numpy as np

class BasicKalmen:

    def __init__(self, x, P, F, H, Q, R):
        self.x = x
        self.P = P
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self):

        self.x = self.F@self.x
        self.P = self.F@self.P@self.F.T + self.Q

        return self.x

    def update(self, z):

        y = z - self.H@self.x
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R)

        self.x = self.x + self.K@self.y

        I = np.eye(self.P.shape[0])
        self.P = (I-K@self.H)@self.P

        return self.x

