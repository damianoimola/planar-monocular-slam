import numpy as np


def back_project(point_2d, camera_matrix):
    """2D point (image frame) to 3D ray (camera frame)"""
    homo_point_2d = [point_2d[0], point_2d[1], 1]
    # r_c = k^-1 * p_homo
    r = np.invert(camera_matrix) * homo_point_2d
    return r

def transform_ray(ray, ):
    """3D ray (camera frame) to 3D ray (world frame)"""
    # r_w =