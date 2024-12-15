import numpy as np
import cv2
from scipy.optimize import least_squares

from load_data import load_camera_file, load_trajectory_file, load_world_file, load_measurement_files

class PlanarMonocularSLAM:
    def __init__(self):
        # ===== DATA LOADING =====
        self.camera_matrix, self.camera_transformation, self.z_near, self.z_far, self.width, self.height = load_camera_file()
        self.trajectory = load_trajectory_file()
        self.world_data = load_world_file()
        self.measurement_data = load_measurement_files()

    def triangulation(self, pose1, pose2, points1, points2):
        # compute projection matrices
        proj1 = self.camera_matrix @ pose1
        proj2 = self.camera_matrix @ pose2
        # triangulate points using opencv
        points_4d = cv2.triangulatePoints(proj1, proj2, points1.T, points2.T)
        # projection from homogeneous
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def run(self):
        print("TRAJECTORY LENGTH:", len(self.trajectory))

        triangulated_points = []
        for i in range(len(self.trajectory)-1):
            print(i)
            # robot pose in world
            current_estimated_pose = self.trajectory[i][2]
            next_estimated_pose = self.trajectory[i+1][2]
            # camera pose in world
            pose1 = current_estimated_pose @ self.camera_transformation
            pose2 = next_estimated_pose @ self.camera_transformation
            # FIXME: self.measurement_data is not ordered by measure id: order it!!!
            # triangulate
            points_3d = self.triangulation(pose1, pose2, self.measurement_data[i], self.measurement_data[i+1])
            triangulated_points.append(points_3d)




PlanarMonocularSLAM()