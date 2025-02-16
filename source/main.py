import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

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

    def plot_results(self, triangulated_points, trajectory_odom, gt_trajectory, refined_params=None):
        """
        Plot:
         - 3D scatter of triangulated points with camera positions.
         - 2D robot trajectory.
         - (Optional) Reprojection comparison for a chosen camera.
        """
        # Convert inputs to arrays.
        triangulated_points = np.array(triangulated_points)
        trajectory_odom = np.array(trajectory_odom)

        # Compute camera positions (apply the camera extrinsics).
        camera_positions_odoms = []
        odoms = []
        for pose in trajectory_odom:
            odoms.append(self.pose_to_matrix(pose)[:3, 3])
            T = self.pose_to_matrix(pose)
            T_cam = self.camera_transformation @ T
            camera_positions_odoms.append(T_cam[:3, 3])
        camera_positions_odoms = np.array(camera_positions_odoms)
        odoms = np.array(odoms)


        camera_positions_gts = []
        gts = []
        for pose in gt_trajectory:
            gts.append(self.pose_to_matrix(pose)[:3, 3])
            T = self.pose_to_matrix(pose)
            T_cam = self.camera_transformation @ T
            camera_positions_gts.append(T_cam[:3, 3])
        camera_positions_gts = np.array(camera_positions_gts)
        gts = np.array(gts)



        fig = plt.figure(figsize=(5, 15), dpi=500)

        # ===== TRAJECTORY =====
        ax1 = fig.add_subplot(311, projection='3d')
        ax1.scatter(odoms[:, 0], odoms[:, 1], odoms[:, 2], s=20, c='red', marker='^', label='Camera Poses (odom)')
        ax1.scatter(gts[:, 0], gts[:, 1], gts[:, 2], s=20, c='blue', marker='^', label='Camera Poses (gts)')
        ax1.set_title("3D Camera Poses")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()

        # ===== POINTS =====
        ax2 = fig.add_subplot(312, projection='3d')
        ax2.scatter(camera_positions_odoms[:, 0], camera_positions_odoms[:, 1], camera_positions_odoms[:, 2], s=20, c='red', marker='^', label='Camera Poses (odom)')
        ax2.scatter(camera_positions_gts[:, 0], camera_positions_gts[:, 1], camera_positions_gts[:, 2], s=20, c='blue', marker='^', label='Camera Poses (gts)')
        ax2.set_title("3D Points")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()

        # ===== POINTS =====
        ax3 = fig.add_subplot(313, projection='3d')
        ax3.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], s=3, c='blue', label='Triangulated Points')
        ax3.set_title("3D Points")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.legend()


        # ax2 = fig.add_subplot(212)
        # # ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', label='Trajectory')
        # ax2.scatter(trajectory[:, 0], trajectory[:, 1], s=1, c='blue', label='Trajectory')
        # ax2.set_title("2D Robot Trajectory")
        # ax2.set_xlabel("X")
        # ax2.set_ylabel("Y")
        # ax2.legend()


        plt.tight_layout()
        plt.show()

    def triangulation(self, pose1, pose2, samples1, samples2):
        # projection matrices
        proj1 = np.array(self.camera_matrix, dtype=np.float64) @ pose1[:3, :]  # Extract 3x4 from 4x4
        proj2 = np.array(self.camera_matrix, dtype=np.float64) @ pose2[:3, :]  # Extract 3x4 from 4x4

        matched_points1 = []
        matched_points2 = []
        # matching points between views
        for i in range(len(samples1)):
            for j in range(len(samples2)):
                if samples1[i].points_id_meas == samples2[j].points_id_meas:
                    matched_points1.append(samples1[i].point)
                    # matched_points2.append(samples2[i].point)
                    matched_points2.append(samples2[j].point)

        # Convert to NumPy arrays of shape (N, 2)
        matched_points1 = np.array(matched_points1, dtype=np.float32)
        matched_points2 = np.array(matched_points2, dtype=np.float32)

        # real triangulation
        points_4d = cv2.triangulatePoints(proj1, proj2, matched_points1.T, matched_points2.T)

        # projection
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def bundle_adjustment(self, triangulated_points, trajectory):
        def residuals(params, num_cameras, num_points, camera_matrix, measurement_data):
            # print("### RESIDUALS")
            # print(params.shape, num_cameras, num_points)

            # Extract camera parameters and 3D points from params
            camera_params = params[:num_cameras * 3].reshape((num_cameras, 3))
            points = params[num_cameras * 3:].reshape((num_points, 3))
            # print(camera_params.shape, points.shape)

            residuals_list = []
            for cam_index, (cam_param, cam_measurements) in enumerate(zip(camera_params, measurement_data)):
                # print(cam_index)
                cam = self.pose_to_matrix(cam_param)

                # Compute camera pose
                # print("cam", cam)
                # R, _ = cv2.Rodrigues(cam[:3, :3])
                R = cam[:3, :3]
                t = cam[:3, 3:].reshape((3, 1))
                # print("R", R)
                # print("t", t)
                # print("CAMERA MATRIX", camera_matrix)
                # print(np.hstack((R, t)))
                proj_matrix = camera_matrix @ np.hstack((R, t))
                # print("PROJ", proj_matrix)

                for measurement in cam_measurements:
                    # print("MEAS", measurement)
                    # Unpack measurement data
                    point_id, feature = measurement
                    point_3d = points[point_id]

                    # print(np.hstack((point_3d, 1)))

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

        # (200, 3) -> (600,)
        print(np.array(trajectory).shape, "->", trajectory_flattened.shape)
        # (19159, 3) -> (57477,)
        print(np.array(triangulated_points).shape, "->", points_flattened.shape)
        # (58077,) = (600,) + (57477,)
        print(initial_params.shape)

        # Perform least-squares optimization
        result = least_squares(
            residuals,
            initial_params,
            args=(num_cameras, num_points, self.camera_matrix, self.measurement_data),
            verbose=2
        )

        return result.x



    # ===== MAIN =====
    def run(self):
        print("===== TRIANGULATION =====")

        print("TRAJECTORY LENGTH:", len(self.trajectory))
        triangulated_points = []
        for i in range(len(self.trajectory)-1):
            # robot pose in world
            current_estimated_pose = self.pose_to_matrix(self.measurement_data[i].odom)
            next_estimated_pose = self.pose_to_matrix(self.measurement_data[i+1].odom)

            # measured points
            current_points = self.measurement_data[i].points
            next_points = self.measurement_data[i+1].points

            # camera pose in world
            pose1 = self.camera_transformation @ current_estimated_pose
            pose2 = self.camera_transformation @ next_estimated_pose

            # triangulate
            points_3d = self.triangulation(pose1, pose2, current_points, next_points)
            triangulated_points.extend(points_3d)
        self.plot_results(triangulated_points, self.trajectory.odoms, self.trajectory.gts)

        print("===== BUNDLE ADJUSTMENT =====")
        # refined_params = self.bundle_adjustment(triangulated_points, np.array(self.trajectory.odoms))



pms = PlanarMonocularSLAM()
pms.run()