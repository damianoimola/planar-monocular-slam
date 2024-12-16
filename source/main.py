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
        unordered_measurement_data = load_measurement_files()
        self.measurement_data = sorted(unordered_measurement_data, key=lambda x:x[0])

    @staticmethod
    def pose_to_matrix(pose):
        x, y, theta = pose
        theta = float(theta)
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s, 0, x],
            [s, c, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

    def triangulation(self, pose1, pose2, samples1, samples2):
        # projection matrices
        proj1 = np.array(self.camera_matrix, dtype=np.float64) @ pose1[:3, :]  # Extract 3x4 from 4x4
        proj2 = np.array(self.camera_matrix, dtype=np.float64) @ pose2[:3, :]  # Extract 3x4 from 4x4

        points1 = []
        points2 = []
        # matching points between views
        for id1, land1, feat1 in samples1:
            for id2, land2, feat2 in samples2:
                if id1 == id2:
                    points1.append(feat1)
                    points2.append(feat2)

        # Convert to NumPy arrays of shape (N, 2)
        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)

        # real triangulation
        points_4d = cv2.triangulatePoints(proj1, proj2, points1.T, points2.T)

        # projection
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def run(self):
        print("TRAJECTORY LENGTH:", len(self.trajectory))

        triangulated_points = []
        for i in range(len(self.trajectory)-1):
            print("ITER", i)
            # robot pose in world
            current_estimated_pose = self.pose_to_matrix(self.measurement_data[i][1])
            next_estimated_pose = self.pose_to_matrix(self.measurement_data[i+1][1])

            # camera pose in world
            pose1 = self.camera_transformation @ current_estimated_pose
            pose2 = self.camera_transformation @ next_estimated_pose

            # triangulate
            points_3d = self.triangulation(pose1, pose2, self.measurement_data[i][3], self.measurement_data[i+1][3])
            triangulated_points.append(points_3d)

        # TODO: Bundle Adjustment


pms = PlanarMonocularSLAM()
pms.run()