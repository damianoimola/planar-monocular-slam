import matplotlib.pyplot as plt

from bundle_adjustment import do_bundle_adjustment
from robust_bundle_adjustment import do_robust_bundle_adjustment
from robust_estimation import RobustMethod
from triangulate_landmarks import triangulate_landmarks
from load_data import load_camera_file, load_trajectory_file, load_world_file, load_measurement_files
from utils import *


class PlanarMonocularSLAM:
    def __init__(self, damping=1, kernel_threshold=1e3, num_iterations=20):
        self.DAMPING = damping
        self.KERNEL_THRESHOLD = kernel_threshold
        self.NUM_ITERATIONS = num_iterations


    def read_data(self):
        print("===== READING DATA =====")
        # ===== CAMERA FILE =====
        self.K, self.camera_transformation, self.z_near, self.z_far, self.image_cols, self.image_rows = load_camera_file()
        self.invK = np.linalg.inv(self.K)

        # ===== TRAJECTORY FILE =====
        self.trajectory = load_trajectory_file()
        self.num_poses = len(self.trajectory)
        self.pose_dim = 3

        # convert poses to homogeneous matrices
        self.XR_guess = np.zeros((3, 3, self.num_poses))
        for i in range(self.num_poses):
            self.XR_guess[:, :, i] = v2t(self.trajectory.get_odom(i))

        self.XR_true = np.zeros((3, 3, self.num_poses))
        for i in range(self.num_poses):
            self.XR_true[:, :, i] = v2t(self.trajectory.get_gt(i))


        # ===== WORLD FILE (GT LANDMARKS) =====
        self.XL_true = np.array(load_world_file()).squeeze(1).transpose()

        # ===== MEASUREMENTS =====
        # === pose-pose measurements
        self.Zr = np.zeros((3, 3, self.num_poses - 1))
        for measurement_num in range(self.num_poses - 1):
            Xi = self.XR_guess[:, :, measurement_num]
            Xj = self.XR_guess[:, :, measurement_num + 1]
            self.Zr[:, :, measurement_num] = np.linalg.inv(Xi) @ Xj

        # === pose-proj measurements
        self.projection_associations = []
        self.Zp = []

        unordered_projection_measurement_data = load_measurement_files()
        self.projection_measurement_data = sorted(unordered_projection_measurement_data, key=lambda x: x.seq)

        for pose_num in range(0, self.num_poses):
            id_landmarks, measurements = self.projection_measurement_data[pose_num].get_ids_and_points()
            for i in range(len(measurements)):
                self.projection_associations.append([pose_num, id_landmarks[i] + 1])
                # self.projection_associations.append([pose_num, id_landmarks[i] - 1])
                # self.projection_associations.append([pose_num, id_landmarks[i]])
                self.Zp.append(measurements[i])
        self.projection_associations = np.array(self.projection_associations).T
        self.Zp = np.array(self.Zp).T

        # ===== LANDMARK INITIALIZATION =====
        self.id_landmarks = np.unique(self.projection_associations[1, :])
        self.num_landmarks = len(self.id_landmarks)
        self.landmark_dim = 3

    def triangulate(self):
        print("###############################################")
        print("###              TRIANGULATION              ###")
        print("###############################################")
        self.XL_guess, self.Zp, self.projection_associations, self.id_landmarks = triangulate_landmarks(self.XR_guess,
                                                                                                        self.Zp,
                                                                                                        self.projection_associations,
                                                                                                        self.id_landmarks,
                                                                                                        self.camera_transformation,
                                                                                                        self.invK,
                                                                                                        self.num_poses,
                                                                                                        self.num_landmarks)


        self.num_landmarks = len(self.id_landmarks)
        self.id_landmarks = self.id_landmarks.reshape(1, -1)

    def pre_optimization(self):
        print("###############################################")
        print("###    Preliminary Landmarks Optimization   ###")
        print("###############################################")
        num_iterations = 5
        block_poses = True
        self.XR_guess, self.XL_guess, self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r, self.H, self.b =(
            do_bundle_adjustment(self.XR_guess,
                                self.XL_guess,
                                self.Zp,
                                self.projection_associations,
                                self.Zr,
                                num_iterations,
                                self.DAMPING,
                                self.KERNEL_THRESHOLD,
                                block_poses,
                                self.num_poses,
                                self.num_landmarks,
                                self.pose_dim,
                                self.landmark_dim,
                                self.K, self.camera_transformation,
                                self.z_near, self.z_far,
                                self.image_rows, self.image_cols))

    def rba(self, robust_method=RobustMethod.HUBER, robust_param=1.0):
        print("###############################################")
        print("###        ROBUST BUNDLE ADJUSTMENT         ###")
        print("###############################################")
        block_poses = False
        self.XR, self.XL, self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r, self.H, self.b = do_robust_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr,
            self.NUM_ITERATIONS, self.DAMPING, self.KERNEL_THRESHOLD, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols,
            robust_method=robust_method, robust_param=robust_param)

        # suppressing useless dimension
        self.chi_stats_p = self.chi_stats_p.squeeze()
        self.num_inliers_p = self.num_inliers_p.squeeze()
        self.chi_stats_r = self.chi_stats_r.squeeze()
        self.num_inliers_r = self.num_inliers_r.squeeze()

    def ba(self):
        print("###############################################")
        print("###            BUNDLE ADJUSTMENT            ###")
        print("###############################################")
        block_poses = False

        self.XR, self.XL, self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r, self.H, self.b = do_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr,
            self.NUM_ITERATIONS, self.DAMPING, self.KERNEL_THRESHOLD, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols)

        # suppressing useless dimension
        self.chi_stats_p = self.chi_stats_p.squeeze()
        self.num_inliers_p = self.num_inliers_p.squeeze()
        self.chi_stats_r = self.chi_stats_r.squeeze()
        self.num_inliers_r = self.num_inliers_r.squeeze()



    def plot(self):
        print("===== PLOTTING =====")
        def _plot_trajectory():
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.title("Poses - Initial scenario")
            plt.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            plt.scatter(self.XR_guess[0:1, 2:, :], self.XR_guess[1:2, 2:, :], color="tomato")
            # plt.scatter(self.XR_true[0, :], self.XR_true[1, :], color="royalblue", marker='*')
            # plt.scatter(self.XR_guess[0, :], self.XR_guess[1, :], color="tomato")
            plt.legend(["ground truth poses", "initial guess poses"])
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.title("Poses - After optimization")
            plt.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            plt.scatter(self.XR[0:1, 2:, :], self.XR[1:2, 2:, :], color="tomato")
            # plt.scatter(self.XR_true[0, :], self.XR_true[1, :], color="royalblue", marker='*')
            # plt.scatter(self.XR[0, :], self.XR[1, :], color="tomato")
            plt.legend(["ground truth poses", "refined guess poses"])
            plt.grid(True)

        def _plot_landmarks():
            plt.figure(2)
            plt.subplot(2, 2, 1)
            plt.title("2D Landmarks - Initial scenario")
            plt.scatter(self.XL_true[0, :], self.XL_true[1, :], color="royalblue", marker='*', s=2)
            plt.scatter(self.XL_guess[0, :], self.XL_guess[1, :], color="tomato", marker='.',s=2)
            plt.legend(["ground truth landmarks", "initial guess landmarks"])
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.title("2D Landmark - After optimization")
            plt.scatter(self.XL_true[0, :], self.XL_true[1, :], color="royalblue", marker='*', s=2)
            plt.scatter(self.XL[0, :], self.XL[1, :], color="tomato", marker='.', s=2)
            plt.legend(["ground truth landmarks", "refined guess landmarks"])
            plt.grid(True)

            plt.subplot(2, 2, 3, projection='3d')
            plt.title("3D Landmarks - Initial scenario")
            plt.scatter(self.XL_true[0, :], self.XL_true[1, :], self.XL_true[2, :], color="royalblue", marker='*')
            plt.scatter(self.XL_guess[0, :], self.XL_guess[1, :], self.XL_guess[2, :], color="tomato", marker='.')
            plt.legend(["ground truth landmarks", "initial guess landmarks"])
            plt.grid(True)

            plt.subplot(2, 2, 4, projection='3d')
            plt.title("3D Landmark - After optimization")
            plt.scatter(self.XL_true[0, :], self.XL_true[1, :], self.XL_true[2, :], color="royalblue", marker='*')
            plt.scatter(self.XL[0, :], self.XL[1, :], self.XL[2, :], color="tomato", marker='.')
            plt.legend(["ground truth landmarks", "refined guess landmarks"])
            plt.grid(True)

        def _plot_chi_and_inliers():
            plt.figure(3)
            # plt.title("Chi Evolution")
            plt.subplot(2, 2, 1)
            plt.plot(self.chi_stats_r, color='darkorange', linewidth=2)
            plt.legend(["Chi poses"])
            plt.grid(True)
            plt.xlabel("Iterations")

            plt.subplot(2, 2, 2)
            plt.plot(self.num_inliers_r, color='hotpink', linewidth=2)
            plt.legend(["num. of inliers"])
            plt.grid(True)
            plt.xlabel("Iterations")

            plt.subplot(2, 2, 3)
            plt.plot(self.chi_stats_p, color='darkorange', linewidth=2)
            plt.legend(["Chi projections"])
            plt.grid(True)
            plt.xlabel("Iterations")

            plt.subplot(2, 2, 4)
            plt.plot(self.num_inliers_p, color='hotpink', linewidth=2)
            plt.legend(["num. of inliers"])
            plt.grid(True)
            plt.xlabel("Iterations")

        plt.style.use('bmh')
        _plot_trajectory()
        _plot_landmarks()
        _plot_chi_and_inliers()
        plt.show()




pms = PlanarMonocularSLAM(damping=1, kernel_threshold=1e3, num_iterations=20)
pms.read_data()
pms.triangulate()
# pms.ba()
pms.rba(robust_method=RobustMethod.CAUCHY, robust_param=4.0)
pms.plot()