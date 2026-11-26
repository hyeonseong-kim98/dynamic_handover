import numpy as np

class KalmanFilter3D:
    def __init__(self, dt, q=1.0, r=1.0):
        # Time step
        self.dt = dt
        self. first_update = True
        # State Transition Matrix
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # Measurement Matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])

        # Initial State Covariance
        self.P = np.eye(6)

        # Process Noise Covariance
        self.Q = q*np.eye(6)

        # Measurement Noise Covariance
        self.R = r*np.eye(3)

        # Initial State
        self.x = np.zeros((6, 1))

    def predict(self):
        # Predict the state
        self.x = self.F @ self.x 

        # Predict the state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if not np.isnan(z).any():
            if self.first_update:
                self.first_update = False
                self.x[0] = z[0].copy()
                self.x[1] = z[1].copy()
                self.x[2] = z[2].copy()
            else:
                # Compute Kalman Gain
                S = self.H @ self.P @ self.H.T + self.R 
                K = self.P @ self.H.T @ np.linalg.inv(S)
                
                # Update the state
                self.x = self.x + K @ (z - self.H @ self.x)

                # Update the state covariance
                self.P = self.P - K @ self.H @ self.P