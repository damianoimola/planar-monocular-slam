import numpy as np
from utils import *

def box_plus(XR, XL, num_poses, num_landmarks, dx, pose_dim, landmark_dim):
    # as in slides of multi-point registration
    for pose_index in range(num_poses):
        pose_matrix_index = pose_matrix_index(pose_index, num_poses, num_landmarks)
        dxr = dx[pose_matrix_index:pose_matrix_index + pose_dim]
        XR[:, :, pose_index] = v2t(dxr) @ XR[:, :, pose_index]

    for landmark_index in range(num_landmarks):
        landmark_matrix_index = landmark_matrix_index(landmark_index, num_poses, num_landmarks)
        dxl = dx[landmark_matrix_index:landmark_matrix_index + landmark_dim, :]
        XL[:, [landmark_index]] += dxl

    return XR, XL


