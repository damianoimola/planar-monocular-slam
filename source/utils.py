import numpy as np

def v2t(v):
    theta = float(v[2])
    c = np.cos(theta, dtype=np.float64)
    s = np.sin(theta, dtype=np.float64)
    M = np.array([[c, -s, v[0]],
                  [s,  c, v[1]],
                  [0,  0,  1]])
    return M

# Global variable equivalent to MATLAB's global R0
R0 = np.array([[0, -1], [1,  0]])

def flatten_matrix_by_columns(M):
    # 3x3 homogeneous transformation matrix to 6d-vector
    v = np.zeros(6)
    v[0:2] = M[:2, 0]
    v[2:4] = M[:2, 1]
    # v[:4] = M[:2, :2].flatten(order='F')  # Column-wise flattening
    v[4:6] = M[:2, 2]
    return v

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])