import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from load_data import load_camera_file, load_trajectory_file, load_world_file, load_measurement_files
from initialize_landmarks import initialize_landmarks
from source.bundle_adjustment import do_bundle_adjustment
from utils import *


class PlanarMonocularSLAM:
    def __init__(self):
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




        print("###############################################")
        print("###              TRIANGULATION              ###")
        print("###############################################")
        self.XL_guess, self.Zp, self.projection_associations, self.id_landmarks = initialize_landmarks(self.XR_guess,
                                                                                                       self.Zp,
                                                                                                       self.projection_associations,
                                                                                                       self.id_landmarks,
                                                                                                       self.camera_transformation,
                                                                                                       self.invK,
                                                                                                       self.num_poses,
                                                                                                       self.num_landmarks)


        self.num_landmarks = len(self.id_landmarks)
        self.id_landmarks = self.id_landmarks.reshape(1, -1)




        print("###############################################")
        print("###    Preliminary Landmarks Optimization   ###")
        print("###############################################")
        num_iterations = 5
        damping = 1
        kernel_threshold = 1e3
        block_poses = True
        self.XR_guess, self.XL_guess, self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r, self.H, self.b =(
            do_bundle_adjustment(self.XR_guess,
                                self.XL_guess,
                                self.Zp,
                                self.projection_associations,
                                self.Zr,
                                num_iterations,
                                damping,
                                kernel_threshold,
                                block_poses,
                                self.num_poses,
                                self.num_landmarks,
                                self.pose_dim,
                                self.landmark_dim,
                                self.K, self.camera_transformation,
                                self.z_near, self.z_far,
                                self.image_rows, self.image_cols))

        print(self.chi_stats_p.shape, self.num_inliers_p.shape, self.chi_stats_r.shape, self.num_inliers_r.shape)
        print(self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r)

        print("###############################################")
        print("##########           LS            ############")
        print("###############################################")
        num_iterations = 20
        block_poses = False
        self.XR, self.XL, self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r, self.H, self.b = do_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr, num_iterations, damping, kernel_threshold, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols)


        self.chi_stats_p = self.chi_stats_p.squeeze()
        self.num_inliers_p = self.num_inliers_p.squeeze()
        self.chi_stats_r = self.chi_stats_r.squeeze()
        self.num_inliers_r = self.num_inliers_r.squeeze()

        print(self.chi_stats_p.shape, self.num_inliers_p.shape, self.chi_stats_r.shape, self.num_inliers_r.shape)
        print(self.chi_stats_p, self.num_inliers_p, self.chi_stats_r, self.num_inliers_r)



        print("===== PLOTTING =====")
        # Plot results
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.title("Poses Initial Guess")
        plt.plot(self.XR_true[0, :], self.XR_true[1, :], 'b*', linewidth=2)
        plt.plot(self.XR_guess[0, :], self.XR_guess[1, :], 'ro', linewidth=2)
        plt.legend(["Poses True", "Guess"])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title("Poses After Optimization")
        plt.plot(self.XR_true[0, :], self.XR_true[1, :], 'b*', linewidth=2)
        plt.plot(self.XR[0, :], self.XR[1, :], 'ro', linewidth=2)
        plt.legend(["Poses True", "Guess"])
        plt.grid()

        plt.figure(2)
        plt.subplot(2, 2, 1)
        plt.title("Landmark Initial Guess")
        plt.plot(self.XL_true[0, :], self.XL_true[1, :], 'b*', linewidth=2)
        plt.plot(self.XL_guess[0, :], self.XL_guess[1, :], 'ro', linewidth=2)
        plt.legend(["Landmark True", "Guess"])
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.title("Landmark After Optimization")
        plt.plot(self.XL_true[0, :], self.XL_true[1, :], 'b*', linewidth=2)
        plt.plot(self.XL[0, :], self.XL[1, :], 'ro', linewidth=2)
        plt.legend(["Landmark True", "Guess"])
        plt.grid()

        plt.subplot(2, 2, 3, projection='3d')
        plt.title("Landmark Initial Guess")
        plt.scatter(self.XL_true[0, :], self.XL_true[1, :], self.XL_true[2, :], c='b', marker='*')
        plt.scatter(self.XL_guess[0, :], self.XL_guess[1, :], self.XL_guess[2, :], c='r', marker='o')
        plt.legend(["Landmark True", "Guess"])
        plt.grid()

        plt.subplot(2, 2, 4, projection='3d')
        plt.title("Landmark After Optimization")
        plt.scatter(self.XL_true[0, :], self.XL_true[1, :], self.XL_true[2, :], c='b', marker='*')
        plt.scatter(self.XL[0, :], self.XL[1, :], self.XL[2, :], c='r', marker='o')
        plt.legend(["Landmark True", "Guess"])
        plt.grid()

        plt.figure(3)
        plt.title("Chi Evolution")
        plt.subplot(2, 2, 1)
        plt.plot(self.chi_stats_r, 'r-', linewidth=2)
        plt.legend(["Chi Poses"])
        plt.grid()
        plt.xlabel("Iterations")

        plt.subplot(2, 2, 2)
        plt.plot(self.num_inliers_r, 'b-', linewidth=2)
        plt.legend(["#Inliers"])
        plt.grid()
        plt.xlabel("Iterations")

        plt.subplot(2, 2, 3)
        plt.plot(self.chi_stats_p, 'r-', linewidth=2)
        plt.legend(["Chi Proj"])
        plt.grid()
        plt.xlabel("Iterations")

        plt.subplot(2, 2, 4)
        plt.plot(self.num_inliers_p, 'b-', linewidth=2)
        plt.legend(["#Inliers"])
        plt.grid()
        plt.xlabel("Iterations")

        plt.show()





pms = PlanarMonocularSLAM()
# pms.run()