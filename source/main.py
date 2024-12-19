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

    def bundle_adjustment(self, triangulated_points, trajectory):
        def residuals(params, points_3d, camera_matrix, measurement_data):
            num_cameras = len(trajectory)

            # as in multi-point registration
            camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))  # 6 DoF per camera
            points = params[num_cameras * 6:].reshape((-1, 3))  # remaining are 3D points

            residuals = []
            for i, (cam_param, cam_measurements) in enumerate(zip(camera_params, measurement_data)):
                # camera pose from parameters
                R, _ = cv2.Rodrigues(cam_param[:3])
                t = cam_param[3:].reshape((3, 1))
                proj_matrix = camera_matrix @ np.hstack((R, t))

                for id, land, feat in cam_measurements:
                    # project 3D point into 2D
                    point_3d = points[id]
                    point_2d_h = proj_matrix @ np.hstack((point_3d, 1))
                    point_2d = point_2d_h[:2] / point_2d_h[2]
                    # reprojection error
                    residuals.append(point_2d - feat)
            return np.concatenate(residuals)

        initial_params = np.hstack((np.array(trajectory).T, np.array(triangulated_points).T))
        result = least_squares(
            residuals,
            initial_params,
            args=(triangulated_points, self.camera_matrix, self.measurement_data),
            verbose=2
        )
        return result.x


    def run(self):
        print("===== TRIANGULATION =====")

        print("TRAJECTORY LENGTH:", len(self.trajectory))
        triangulated_points = []
        for i in range(len(self.trajectory)-1):
            # robot pose in world
            current_estimated_pose = self.pose_to_matrix(self.measurement_data[i][1])
            next_estimated_pose = self.pose_to_matrix(self.measurement_data[i+1][1])

            # camera pose in world
            pose1 = self.camera_transformation @ current_estimated_pose
            pose2 = self.camera_transformation @ next_estimated_pose

            # triangulate
            points_3d = self.triangulation(pose1, pose2, self.measurement_data[i][3], self.measurement_data[i+1][3])
            triangulated_points.extend(points_3d)

        print("===== BUNDLE ADJUSTMENT =====")
        refined_params = self.bundle_adjustment(triangulated_points, np.array(self.trajectory.odoms))



pms = PlanarMonocularSLAM()
pms.run()