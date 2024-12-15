import numpy as np
import cv2
from scipy.optimize import least_squares

from load_data import load_camera_file, load_trajectory_file, load_world_file, load_measurement_files


def order_by_measurement_id(measurement):
    return measurement[0]

class PlanarMonocularSLAM:
    def __init__(self):
        # ===== DATA LOADING =====
        self.camera_matrix, self.camera_transformation, self.z_near, self.z_far, self.width, self.height = load_camera_file()
        self.trajectory = load_trajectory_file()
        self.world_data = load_world_file()
        unordered_measurement_data = load_measurement_files()
        self.measurement_data = sorted(unordered_measurement_data, key=lambda x:x[0])

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
            print("ITER", i)
            # robot pose in world
            # current_estimated_pose = self.trajectory[i][2]
            # next_estimated_pose = self.trajectory[i+1][2]
            current_estimated_pose = self.measurement_data[i][1]
            next_estimated_pose = self.measurement_data[i+1][1]

            # FIXME: mmh, double check if this is the right way
            print(np.array(current_estimated_pose, dtype=np.float32)) # 3x3
            print(np.array(self.camera_transformation, dtype=np.float32)) # 4x4

            # camera pose in world
            pose1 = np.array(current_estimated_pose, dtype=np.float32) @ np.array(self.camera_transformation, dtype=np.float32)
            pose2 = np.array(next_estimated_pose, dtype=np.float32) @ np.array(self.camera_transformation, dtype=np.float32)

            # triangulate
            # index [i] is for the i-th measurement
            # index [i][3] retrieves all samples of the i-th measurements
            # slice [i][3][2] retrieves only points from samples of the i-th measurements
            points_3d = self.triangulation(pose1, pose2, self.measurement_data[i][3][2], self.measurement_data[i+1][3][2])
            triangulated_points.append(points_3d)




pms = PlanarMonocularSLAM()
pms.run()