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
        self.measurement_data = sorted(unordered_measurement_data, key=lambda x:x.seq)

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
        for i in range(len(samples1)):
            for j in range(len(samples2)):
                if samples1[i].points_id_meas == samples2[j].points_id_meas:
                    points1.append(samples1[i].point)
                    points2.append(samples2[i].point)

        # Convert to NumPy arrays of shape (N, 2)
        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)

        # real triangulation
        points_4d = cv2.triangulatePoints(proj1, proj2, points1.T, points2.T)

        # projection
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def bundle_adjustment(self, triangulated_points, trajectory):
        def residuals(params, num_cameras, num_points, camera_matrix, measurement_data):
            print(params.shape, num_cameras, num_points)

            # Extract camera parameters and 3D points from params
            camera_params = params[:num_cameras * 3].reshape((num_cameras, 3))
            points = params[num_cameras * 3:].reshape((num_points, 3))
            print(camera_params.shape, points.shape)

            residuals_list = []
            for cam_index, (cam_param, cam_measurements) in enumerate(zip(camera_params, measurement_data)):
                cam = self.pose_to_matrix(cam_param)

                # Compute camera pose
                print(cam_index, cam)
                R, _ = cv2.Rodrigues(cam[:3, :3])
                t = cam[:3, 3:].reshape((3, 1))
                proj_matrix = camera_matrix @ np.hstack((R, t))

                for measurement in cam_measurements:
                    print(cam_measurements)
                    # Unpack measurement data
                    point_id, feature = measurement
                    point_3d = points[point_id]

                    # Project 3D point to 2D
                    point_2d_h = proj_matrix @ np.hstack((point_3d, 1))
                    point_2d = point_2d_h[:2] / point_2d_h[2]

                    # Compute reprojection error
                    residuals_list.extend(point_2d - feature)

            return np.array(residuals_list)

        # Number of cameras and points
        num_cameras = len(trajectory)
        num_points = len(triangulated_points)

        # Flatten trajectory and triangulated points into initial params
        trajectory_flattened = np.array(trajectory).reshape(-1)
        points_flattened = np.array(triangulated_points).reshape(-1)
        initial_params = np.hstack((trajectory_flattened, points_flattened))

        print(np.array(trajectory).shape, "->", trajectory_flattened.shape)
        print(np.array(triangulated_points).shape, "->", points_flattened.shape)
        print(initial_params.shape)

        # Perform least-squares optimization
        result = least_squares(
            residuals,
            initial_params,
            args=(num_cameras, num_points, self.camera_matrix, self.measurement_data),
            verbose=2
        )

        return result.x


    def run(self):
        print("===== TRIANGULATION =====")

        print("TRAJECTORY LENGTH:", len(self.trajectory))
        triangulated_points = []
        for i in range(len(self.trajectory)-1):
            # robot pose in world
            current_estimated_pose = self.pose_to_matrix(self.measurement_data[i].odom)
            next_estimated_pose = self.pose_to_matrix(self.measurement_data[i+1].odom)

            # camera pose in world
            pose1 = self.camera_transformation @ current_estimated_pose
            pose2 = self.camera_transformation @ next_estimated_pose

            # triangulate
            points_3d = self.triangulation(pose1, pose2, self.measurement_data[i].points, self.measurement_data[i+1].points)
            triangulated_points.extend(points_3d)

        print("===== BUNDLE ADJUSTMENT =====")
        refined_params = self.bundle_adjustment(triangulated_points, np.array(self.trajectory.odoms))



pms = PlanarMonocularSLAM()
pms.run()