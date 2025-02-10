# the code of this file is adapted from what we've seen during lectures of multi-point registration:
#   https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/blob/main/source/octave/25_multi_point_registration/multi_ICP_3d.m?ref_type=heads

from utils import *

from bundle_adjustment_indices import pose_matrix_index, landmark_matrix_index
from bundle_adjustment_eaj import projection_error_and_jacobian, pose_error_and_jacobian



def build_linear_system_poses(XR, XL, Zr, kernel_threshold, num_poses, num_landmarks, pose_dim, landmark_dim):

    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    H = np.zeros((system_size, system_size))
    b = np.zeros(system_size)
    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(Zr.shape[2]):
        Omega = np.eye(6)
        Omega[:4, :4] *= 1e3

        Z = Zr[:, :, measurement_num]
        Xi = XR[:, :, measurement_num]
        Xj = XR[:, :, measurement_num + 1]

        e, Ji, Jj = pose_error_and_jacobian(Xi, Xj, Z)
        chi = e.T @ Omega @ e

        if chi > kernel_threshold:
            Omega *= np.sqrt(kernel_threshold / chi)
            chi = kernel_threshold
        else:
            num_inliers += 1

        chi_tot += chi

        pose_i_matrix_index = pose_matrix_index(measurement_num)
        pose_j_matrix_index = pose_matrix_index(measurement_num + 1)

        # ===== H MATRIX =====
        H[pose_i_matrix_index:pose_i_matrix_index + pose_dim,
        pose_i_matrix_index:pose_i_matrix_index + pose_dim] += Ji.T @ Omega @ Ji

        H[pose_i_matrix_index:pose_i_matrix_index + pose_dim,
        pose_j_matrix_index:pose_j_matrix_index + pose_dim] += Ji.T @ Omega @ Jj

        H[pose_j_matrix_index:pose_j_matrix_index + pose_dim,
        pose_i_matrix_index:pose_i_matrix_index + pose_dim] += Jj.T @ Omega @ Ji

        H[pose_j_matrix_index:pose_j_matrix_index + pose_dim,
        pose_j_matrix_index:pose_j_matrix_index + pose_dim] += Jj.T @ Omega @ Jj

        # ===== b VECTOR =====
        b[pose_i_matrix_index:pose_i_matrix_index + pose_dim] += Ji.T @ Omega @ e
        b[pose_j_matrix_index:pose_j_matrix_index + pose_dim] += Jj.T @ Omega @ e

    return H, b, chi_tot, num_inliers



def build_linear_system_projections(XR, XL, Zp, associations, kernel_threshold, num_poses, num_landmarks, pose_dim, landmark_dim,
                                    K, camera_transformation, z_near, z_far, image_rows, image_cols):

    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    H = np.zeros((system_size, system_size))
    b = np.zeros(system_size)
    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(Zp.shape[1]):
        pose_index = associations[0, measurement_num]
        landmark_index = associations[1, measurement_num]
        z = Zp[:, measurement_num]
        Xr = XR[:, :, pose_index]
        Xl = XL[:, landmark_index]

        is_valid, e, Jr, Jl = projection_error_and_jacobian(Xr, Xl, z, K, camera_transformation, z_near, z_far, image_rows, image_cols)
        if not is_valid:
            continue

        chi = e.T @ e
        if chi > kernel_threshold:
            e *= np.sqrt(kernel_threshold / chi)
            chi = kernel_threshold
        else:
            num_inliers += 1

        chi_tot += chi

        pose_matrix_idx = pose_matrix_index(pose_index, pose_dim)
        landmark_matrix_idx = landmark_matrix_index(landmark_index, pose_dim, landmark_dim, num_poses)

        # ===== H MATRIX =====
        H[pose_matrix_idx:pose_matrix_idx + pose_dim,
        pose_matrix_idx:pose_matrix_idx + pose_dim] += Jr.T @ Jr

        H[pose_matrix_idx:pose_matrix_idx + pose_dim,
        landmark_matrix_idx:landmark_matrix_idx + landmark_dim] += Jr.T @ Jl

        H[landmark_matrix_idx:landmark_matrix_idx + landmark_dim,
        landmark_matrix_idx:landmark_matrix_idx + landmark_dim] += Jl.T @ Jl

        H[landmark_matrix_idx:landmark_matrix_idx + landmark_dim,
        pose_matrix_idx:pose_matrix_idx + pose_dim] += Jl.T @ Jr

        # ===== b VECTOR =====
        b[pose_matrix_idx:pose_matrix_idx + pose_dim] += Jr.T @ e
        b[landmark_matrix_idx:landmark_matrix_idx + landmark_dim] += Jl.T @ e

    return H, b, chi_tot, num_inliers
