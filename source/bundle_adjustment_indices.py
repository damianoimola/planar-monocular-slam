# the code of this file is adapted from what we've seen during lectures of multi-point registration:
#   https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/blob/main/source/octave/25_multi_point_registration/multi_ICP_3d.m?ref_type=heads


def pose_matrix_index(pose_index, pose_dim):
    return 1 + (pose_index - 1) * pose_dim


def landmark_matrix_index(landmark_index, pose_dim, landmark_dim, num_poses):
    return 1 + (num_poses * pose_dim) + (landmark_index - 1) * landmark_dim