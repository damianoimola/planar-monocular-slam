# the code of this file is adapted from what we've seen during lectures of multi-point registration:
#   https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/blob/main/source/octave/25_multi_point_registration/multi_ICP_3d.m?ref_type=heads
from utils import *
from tqdm import tqdm

from robust_bundle_adjustment_lin_sys import build_robust_linear_system_projections, build_robust_linear_system_poses
from bundle_adjustment_indices import landmark_matrix_index, pose_matrix_index
from robust_estimation import robust_weight, RobustMethod


def box_plus(XR, XL, dx, num_poses, num_landmarks, pose_dim, landmark_dim):
    XR_copy = XR.copy()
    XL_copy = XL.copy()

    # as in slides of multi-point registration
    for pose_index in range(num_poses):
        pose_matrix_idx = pose_matrix_index(pose_index, pose_dim, num_poses)
        dxr = dx[pose_matrix_idx:pose_matrix_idx + pose_dim]
        XR_copy[:, :, pose_index] = v2t(dxr) @ XR[:, :, pose_index]

    for landmark_index in range(num_landmarks):
        landmark_matrix_idx = landmark_matrix_index(landmark_index, pose_dim, landmark_dim, num_poses, num_landmarks)
        dxl = dx[landmark_matrix_idx:landmark_matrix_idx + landmark_dim]
        # XL[:, [landmark_index]] += dxl
        XL_copy[:, landmark_index] += dxl

    return XR_copy, XL_copy




# in the code doTotalLs
def do_robust_bundle_adjustment(XR, XL, Zp, projection_associations,
                         Zr, num_iterations, damping, kernel_threshold,
                         block_poses, num_poses, num_landmarks, pose_dim, landmark_dim,
                         K, camera_transformation, z_near, z_far, image_rows, image_cols,
                         robust_method=RobustMethod.NONE, robust_param=1.0):
    """
        NOTE: a good starting value for `robust_param` value is 1.0 for
        huber and [2.0, 4.0] for Cauchy, Tukey
    """
    chi_stats_p = np.zeros((1, num_iterations))
    num_inliers_p = np.zeros((1, num_iterations))
    chi_stats_r = np.zeros((1, num_iterations))
    num_inliers_r = np.zeros((1, num_iterations))

    system_size = pose_dim * num_poses + landmark_dim * num_landmarks

    for iteration in tqdm(range(num_iterations)):
        H = np.zeros((system_size, system_size))
        b = np.zeros(system_size)

        # ===== LINEARIZE PROJECTIONS =====
        H_projections, b_projections, chi_, num_inliers_, _ = build_robust_linear_system_projections(
            XR, XL, Zp, projection_associations, kernel_threshold,
            num_poses, num_landmarks, pose_dim, landmark_dim,
            K, camera_transformation, z_near, z_far, image_rows, image_cols,
            robust_method, robust_param
        )

        H += H_projections
        b += b_projections
        chi_stats_p[:, iteration] += chi_
        num_inliers_p[:, iteration] = num_inliers_

        if not block_poses:
            # ===== LINEARIZE POSES =====
            H_poses, b_poses, chi_, num_inliers_, _ = build_robust_linear_system_poses(
                XR, XL, Zr, kernel_threshold,
                num_poses, num_landmarks, pose_dim, landmark_dim,
                robust_method, robust_param
            )

            H += H_poses
            b += b_poses

            chi_stats_r[:, iteration] += chi_
            num_inliers_r[:, iteration] = num_inliers_

        H += np.eye(system_size) * damping
        dx = np.zeros(system_size)

        if block_poses:
            dx[pose_dim * num_poses:] = -np.linalg.solve(H[pose_dim * num_poses:, pose_dim * num_poses:],
                                                         b[pose_dim * num_poses:])
        else:
            dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:])

        XR, XL = box_plus(XR, XL, dx, num_poses, num_landmarks, pose_dim, landmark_dim)

    return XR, XL, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b