# the code of this file is adapted from what we've seen during lectures of multi-point registration:
#   https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/blob/main/source/octave/25_multi_point_registration/multi_ICP_3d.m?ref_type=heads

import numpy as np

from utils import *

def pose_error_and_jacobian(Xi, Xj, Z, R0):
    Ri = Xi[:2, :2]
    Rj = Xj[:2, :2]
    ti = Xi[:2, 2]
    tj = Xj[:2, 2]
    tij = tj - ti
    Ri_transposed = Ri.T

    Ji = np.zeros((6, 3))
    Jj = np.zeros((6, 3))

    Jj[4:6, :2] = Ri_transposed
    Jj[:4, 2] = (Ri_transposed @ R0 @ Rj.T).flatten()
    # Jj[:4, 2] = Ri_transposed @ R0 @ Rj.T.flatten()
    Jj[4:6, 2] = -Ri_transposed @ R0 @ tj
    Ji = -Jj

    Z_hat = np.eye(3)
    Z_hat[:2, :2] = Ri_transposed @ Rj
    Z_hat[:2, 2] = Ri_transposed @ tij
    # e = (Z_hat - Z).flatten()
    e=flatten_matrix_by_columns(Z_hat-Z)

    return e, Ji, Jj



def projection_error_and_jacobian(Xr, Xl, z, K, camera_transformation, z_near, z_far, image_rows, image_cols):
    is_valid = False
    e = np.zeros(2)
    Jr = np.zeros((2, 3))
    Jl = np.zeros((2, 3))

    X_robot = np.eye(4)
    X_robot[:2, :2] = Xr[:2, :2]
    X_robot[:2, 3] = Xr[:2, 2]

    iR_cam = camera_transformation[:3, :3].T
    it_cam = -iR_cam @ camera_transformation[:3, 3]

    iR = X_robot[:3, :3].T
    it = -iR @ X_robot[:3, 3]

    pw = iR_cam @ (iR @ Xl + it) + it_cam
    if pw[2] < z_near:
        return is_valid, e, Jr, Jl

    Jwr = np.zeros((3, 3))
    Jwr[:3, :2] = -iR_cam @ iR @ np.array([[1, 0], [0, 1], [0, 0]])
    Jwr[:3, 2] = iR_cam @ iR @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]) @ Xl
    Jwl = iR_cam @ iR

    p_cam = K @ pw
    iz = 1.0 / p_cam[2]
    z_hat = p_cam[:2] * iz

    if not (0 <= z_hat[0] <= image_cols and 0 <= z_hat[1] <= image_rows):
        return is_valid, e, Jr, Jl

    iz2 = iz * iz
    Jp = np.array([[iz, 0, -p_cam[0] * iz2], [0, iz, -p_cam[1] * iz2]])

    e = z_hat - z
    Jr = Jp @ K @ Jwr
    Jl = Jp @ K @ Jwl
    is_valid = True

    return is_valid, e, Jr, Jl