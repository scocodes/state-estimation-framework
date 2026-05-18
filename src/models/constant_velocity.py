import numpy as np 

def constant_velocity_model(dt):
    F = np.array([[1, 0 , dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    return F, H
    
