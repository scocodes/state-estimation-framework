import numpy as np

class BasicKalmen:

    def two_state(px, py, vx, vy):

        x0 = np.array([[px, py, vx, vy]]).T
        dt = 1
        
        F = np.array([[1, 0, dt, 0], 
                    [0, 1, 0, dt], 
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]])
        
        x_pred = F@x0

        P = np.eye(4)*100
        Q = np.eye(4)*0.01

        H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
        
        
        p_pred = F@P@F.T + Q

        sensor_measurement = np.array([[12, 1.5]]).T
        y = sensor_measurement - H@x_pred

        R = np.array([[4]])

        K = P@H.T@np.linalg.inv(H@P@H.T + R)

        x_new = x_pred + K@y

        print(x_new)

bk = BasicKalmen()
bk.two_state()
