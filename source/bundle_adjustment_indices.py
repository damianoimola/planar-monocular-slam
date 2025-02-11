# the code of this file is adapted from what we've seen during lectures of multi-point registration:
#   https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/blob/main/source/octave/25_multi_point_registration/multi_ICP_3d.m?ref_type=heads


def pose_matrix_index(pose_index, pose_dim, num_poses):
    if pose_index>num_poses:
        return -1
    return pose_index * pose_dim
    # return 1 + (pose_index-1)*pose_dim;


def landmark_matrix_index(landmark_index, pose_dim, landmark_dim, num_poses, num_landmarks):
    if landmark_index>num_landmarks:
        return -1
    return (num_poses * pose_dim) + landmark_index * landmark_dim