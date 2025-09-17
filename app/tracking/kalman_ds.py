# -*- coding: utf-8 -*-
import numpy as np

# DeepSORT-style Kalman (CV model)
# State: x = [u, v, gamma, h, u_dot, v_dot, gamma_dot, h_dot]^T
# Measurement: z = [u, v, gamma, h]^T

def _xyxy_to_z(box_xyxy: np.ndarray) -> np.ndarray:
    x1,y1,x2,y2 = [float(v) for v in box_xyxy]
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    gamma = w / h
    return np.array([u, v, gamma, h], dtype=np.float32).reshape(4,1)

def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
    u, v, gamma, h = [float(z[i,0]) for i in range(4)]
    h = max(1e-3, h)
    w = max(1e-3, gamma * h)
    x1 = u - w*0.5; x2 = u + w*0.5
    y1 = v - h*0.5; y2 = v + h*0.5
    return np.array([x1,y1,x2,y2], dtype=np.float32)

class KalmanDS:
    """DeepSORT CV Kalman with height-scaled noise (as in the paper/impl)."""
    ndim, mdim = 8, 4

    def __init__(self, dt: float = 1/30.0,
                 std_weight_pos: float = 1/20.0,
                 std_weight_vel: float = 1/160.0):
        self.dt = float(max(dt, 1e-6))
        self.std_weight_pos = float(std_weight_pos)
        self.std_weight_vel = float(std_weight_vel)

        self.x = np.zeros((self.ndim,1), np.float32)
        self.P = np.eye(self.ndim, dtype=np.float32) * 1e3

        self.F = np.eye(self.ndim, dtype=np.float32)
        self.F[0,4] = self.dt  # u <- u + u_dot*dt
        self.F[1,5] = self.dt  # v
        self.F[2,6] = self.dt  # gamma
        self.F[3,7] = self.dt  # h

        self.H = np.zeros((self.mdim,self.ndim), np.float32)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = self.H[3,3] = 1.0

    def _Q(self, h: float) -> np.ndarray:
        h = float(max(1e-3, h))
        std_pos = np.array([
            self.std_weight_pos*h,  # u
            self.std_weight_pos*h,  # v
            1e-2,                   # gamma
            self.std_weight_pos*h   # h
        ], dtype=np.float32)
        std_vel = np.array([
            self.std_weight_vel*h,  # u_dot
            self.std_weight_vel*h,  # v_dot
            1e-3,                   # gamma_dot
            self.std_weight_vel*h   # h_dot
        ], dtype=np.float32)
        q = np.concatenate([std_pos, std_vel])**2
        return np.diag(q).astype(np.float32)

    def _R(self, h: float) -> np.ndarray:
        h = float(max(1e-3, h))
        std_meas = np.array([
            self.std_weight_pos*h,  # u
            self.std_weight_pos*h,  # v
            1e-2,                   # gamma
            self.std_weight_pos*h   # h
        ], dtype=np.float32)
        r = (std_meas**2)
        return np.diag(r).astype(np.float32)

    def init_from_xyxy(self, box_xyxy: np.ndarray):
        z = _xyxy_to_z(box_xyxy)
        self.x[:4,0] = z[:,0]
        self.x[4:,0] = 0.0
        self.P = np.eye(self.ndim, dtype=np.float32) * 10.0

    def predict(self):
        h = float(max(1e-3, self.x[3,0]))
        Q = self._Q(h)
        self.x = (self.F @ self.x).astype(np.float32)
        self.P = (self.F @ self.P @ self.F.T + Q).astype(np.float32)
        return self.get_xyxy()

    def update_xyxy(self, meas_box_xyxy: np.ndarray):
        z = _xyxy_to_z(meas_box_xyxy)
        R = self._R(float(z[3,0]))
        S = (self.H @ self.P @ self.H.T + R).astype(np.float32)
        K = (self.P @ self.H.T @ np.linalg.inv(S)).astype(np.float32)
        y = (z - self.H @ self.x).astype(np.float32)
        self.x = (self.x + K @ y).astype(np.float32)
        I = np.eye(self.ndim, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def maha_xyxy(self, meas_box_xyxy: np.ndarray) -> float:
        z = _xyxy_to_z(meas_box_xyxy)
        R = self._R(float(z[3,0]))
        S = (self.H @ self.P @ self.H.T + R).astype(np.float32)
        y = (z - self.H @ self.x).astype(np.float32)
        try:
            d = float((y.T @ np.linalg.inv(S) @ y)[0,0])
        except np.linalg.LinAlgError:
            d = 1e9
        return d

    def get_xyxy(self) -> np.ndarray:
        return _z_to_xyxy(self.x[:4,:])
