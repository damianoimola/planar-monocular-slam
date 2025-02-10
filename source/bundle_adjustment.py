# the code of this file is adapted from what we've seen during lectures of multi-point registration:
#   https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/blob/main/source/octave/25_multi_point_registration/multi_ICP_3d.m?ref_type=heads

from utils import *

from bundle_adjustment_lin_sys import build_linear_system_poses, build_linear_system_projections
from bundle_adjustment_indices import landmark_matrix_index, pose_matrix_index



def box_plus(XR, XL, dx, num_poses, num_landmarks, pose_dim, landmark_dim):
    # as in slides of multi-point registration
    for pose_index in range(num_poses):
        pose_matrix_idx = pose_matrix_index(pose_index, num_poses, num_landmarks)
        dxr = dx[pose_matrix_idx:pose_matrix_idx + pose_dim]
        XR[:, :, pose_index] = v2t(dxr) @ XR[:, :, pose_index]

    for landmark_index in range(num_landmarks):
        landmark_matrix_idx = landmark_matrix_index(landmark_index, num_poses, num_landmarks)
        dxl = dx[landmark_matrix_idx:landmark_matrix_idx + landmark_dim, :]
        XL[:, [landmark_index]] += dxl

    return XR, XL




# in the code doTotalLs
def do_bundle_adjustment(XR, XL, Zp, projection_associations,
                         Zr, num_iterations, damping, kernel_threshold,
                         block_poses, num_poses, num_landmarks, pose_dim, landmark_dim,
                         K, camera_transformation, z_near, z_far, image_rows, image_cols):
    # global num_poses, num_landmarks, pose_dim, landmark_dim

    chi_stats_p = np.zeros(num_iterations)
    num_inliers_p = np.zeros(num_iterations)
    chi_stats_r = np.zeros(num_iterations)
    num_inliers_r = np.zeros(num_iterations)

    system_size = pose_dim * num_poses + landmark_dim * num_landmarks

    for iteration in range(num_iterations):
        H = np.zeros((system_size, system_size))
        b = np.zeros(system_size)

        # ===== LINEARIZE PROJECTIONS =====
        H_projections, b_projections, chi_, num_inliers_ = build_linear_system_projections(XR, XL, Zp, projection_associations, kernel_threshold,
                                                                                           num_poses, num_landmarks, pose_dim, landmark_dim,
                                                                                           K, camera_transformation, z_near, z_far, image_rows, image_cols)
        chi_stats_p[iteration] += chi_
        num_inliers_p[iteration] = num_inliers_
        H += H_projections
        b += b_projections

        if not block_poses:
            # ===== LINEARIZE PROJECTIONS =====
            H_poses, b_poses, chi_, num_inliers_ = build_linear_system_poses(XR, XL, Zr, kernel_threshold,
                                                                             num_poses, num_landmarks, pose_dim, landmark_dim)
            chi_stats_r[iteration] += chi_
            num_inliers_r[iteration] = num_inliers_
            H += H_poses
            b += b_poses

        H += np.eye(system_size) * damping
        dx = np.zeros(system_size)

        if block_poses:
            dx[pose_dim * num_poses:] = -np.linalg.solve(H[pose_dim * num_poses:, pose_dim * num_poses:],
                                                         b[pose_dim * num_poses:])
        else:
            dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:])

        XR, XL = box_plus(XR, XL, dx, num_poses, num_landmarks, pose_dim, landmark_dim)

    return XR, XL, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b
