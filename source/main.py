import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse

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

    # ===== TRIANGULATION =====
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

    # ===== PRE OPTIMIZATION (ONLY-LANDMARKS BA)  (~20 sec)=====
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

    # ===== ROBUST BUNDLE ADJUSTMENT  (~1 min) =====
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

    # ===== BUNDLE ADJUSTMENT  (~1 min) =====
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

    # ===== PLOTTING =====
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

    # ===== BUNDLE ADJUSTMENT + ROBUST BUNDLE ADJUSTMENTS + PLOT  (~4.5 min) =====
    def comparison(self):
        print("###############################################")
        print("###            BUNDLE ADJUSTMENT            ###")
        print("###############################################")
        block_poses = False

        self.XR_ba, self.XL_ba, self.chi_stats_p_ba, self.num_inliers_p_ba, self.chi_stats_r_ba, self.num_inliers_r_ba, self.H_ba, self.b_ba = do_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr,
            self.NUM_ITERATIONS, self.DAMPING, self.KERNEL_THRESHOLD, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols)

        # suppressing useless dimension
        self.chi_stats_p_ba = self.chi_stats_p_ba.squeeze()
        self.num_inliers_p_ba = self.num_inliers_p_ba.squeeze()
        self.chi_stats_r_ba = self.chi_stats_r_ba.squeeze()
        self.num_inliers_r_ba = self.num_inliers_r_ba.squeeze()


        print("###############################################")
        print("###       (CAUCHY) BUNDLE ADJUSTMENT        ###")
        print("###############################################")
        block_poses = False
        self.XR_cau, self.XL_cau, self.chi_stats_p_cau, self.num_inliers_p_cau, self.chi_stats_r_cau, self.num_inliers_r_cau, self.H_cau, self.b_cau = do_robust_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr,
            self.NUM_ITERATIONS, self.DAMPING, self.KERNEL_THRESHOLD, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols,
            robust_method=RobustMethod.CAUCHY, robust_param=3.0)

        # suppressing useless dimension
        self.chi_stats_p_cau = self.chi_stats_p_cau.squeeze()
        self.num_inliers_p_cau = self.num_inliers_p_cau.squeeze()
        self.chi_stats_r_cau = self.chi_stats_r_cau.squeeze()
        self.num_inliers_r_cau = self.num_inliers_r_cau.squeeze()


        print("###############################################")
        print("###       (HUBER) BUNDLE ADJUSTMENT         ###")
        print("###############################################")
        block_poses = False
        self.XR_hu, self.XL_hu, self.chi_stats_p_hu, self.num_inliers_p_hu, self.chi_stats_r_hu, self.num_inliers_r_hu, self.H_hu, self.b_hu = do_robust_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr,
            self.NUM_ITERATIONS, self.DAMPING, self.KERNEL_THRESHOLD, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols,
            robust_method=RobustMethod.HUBER, robust_param=1.0)

        # suppressing useless dimension
        self.chi_stats_p_hu = self.chi_stats_p_hu.squeeze()
        self.num_inliers_p_hu = self.num_inliers_p_hu.squeeze()
        self.chi_stats_r_hu = self.chi_stats_r_hu.squeeze()
        self.num_inliers_r_hu = self.num_inliers_r_hu.squeeze()



        print("###############################################")
        print("###       (TUKEY) BUNDLE ADJUSTMENT         ###")
        print("###############################################")
        block_poses = False
        self.XR_tu, self.XL_tu, self.chi_stats_p_tu, self.num_inliers_p_tu, self.chi_stats_r_tu, self.num_inliers_r_tu, self.H_tu, self.b_tu = do_robust_bundle_adjustment(
            self.XR_guess, self.XL_guess, self.Zp, self.projection_associations, self.Zr,
            self.NUM_ITERATIONS, self.DAMPING, self.KERNEL_THRESHOLD, block_poses,
            self.num_poses, self.num_landmarks, self.pose_dim, self.landmark_dim,
            self.K, self.camera_transformation, self.z_near, self.z_far, self.image_rows, self.image_cols,
            robust_method=RobustMethod.TUKEY, robust_param=5.0)

        # suppressing useless dimension
        self.chi_stats_p_tu = self.chi_stats_p_tu.squeeze()
        self.num_inliers_p_tu = self.num_inliers_p_tu.squeeze()
        self.chi_stats_r_tu = self.chi_stats_r_tu.squeeze()
        self.num_inliers_r_tu = self.num_inliers_r_tu.squeeze()

        print("===== PLOTTING =====")

        def _plot_trajectory():
            plt.figure(1)

            plt.subplots_adjust(top=0.95,
                                bottom=0.05,
                                left=0.05,
                                right=0.95,
                                hspace=0.2,
                                wspace=0.2)

            fig = plt.gcf()

            gs = gridspec.GridSpec(nrows=3, ncols=4, figure=fig)

            ax1 = fig.add_subplot(gs[0, 1:3])
            ax1.set_title("Poses\nInitial scenario", fontsize=10)
            ax1.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            ax1.scatter(self.XR_guess[0:1, 2:, :], self.XR_guess[1:2, 2:, :], color="tomato")
            ax1.legend(["ground truth poses", "initial guess poses"])
            ax1.grid(True)



            ax2 = fig.add_subplot(gs[1, 0:2])
            ax2.set_title("Poses\nBundle Adjustment", fontsize=10)
            ax2.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            ax2.scatter(self.XR_ba[0:1, 2:, :], self.XR_ba[1:2, 2:, :], color="tomato")
            ax2.legend(["ground truth poses", "refined guess poses"])
            ax2.grid(True)


            ax3 = fig.add_subplot(gs[1, 2:4])
            ax3.set_title("Poses\nBundle Adjustment (Huber, c=1.0)", fontsize=10)
            ax3.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            ax3.scatter(self.XR_hu[0:1, 2:, :], self.XR_hu[1:2, 2:, :], color="tomato")
            ax3.legend(["ground truth poses", "refined guess poses"])
            ax3.grid(True)


            ax4 = fig.add_subplot(gs[2, 0:2])
            ax4.set_title("Poses\nBundle Adjustment (Cauchy, c=3.0)", fontsize=10)
            ax4.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            ax4.scatter(self.XR_cau[0:1, 2:, :], self.XR_cau[1:2, 2:, :], color="tomato")
            ax4.legend(["ground truth poses", "refined guess poses"])
            ax4.grid(True)


            ax5 = fig.add_subplot(gs[2, 2:4])
            ax5.set_title("Poses\nBundle Adjustment (Tukey, c=5.0)", fontsize=10)
            ax5.scatter(self.XR_true[0:1, 2:, :], self.XR_true[1:2, 2:, :], color="royalblue", marker='*')
            ax5.scatter(self.XR_tu[0:1, 2:, :], self.XR_tu[1:2, 2:, :], color="tomato")
            ax5.legend(["ground truth poses", "refined guess poses"])
            ax5.grid(True)

        def _plot_chi_and_inliers():
            plt.figure(3)
            plt.subplots_adjust(top=0.95,
                                bottom=0.05,
                                left=0.02,
                                right=0.98,
                                hspace=0.28,
                                wspace=0.1)

            plt.subplot(2, 2, 1)
            plt.title("Chi evolution\nPoses")
            plt.plot(self.chi_stats_r_ba, linewidth=2)
            plt.plot(self.chi_stats_r_hu, linewidth=2)
            plt.plot(self.chi_stats_r_tu, linewidth=2)
            plt.plot(self.chi_stats_r_cau, linewidth=2)
            plt.legend(["classic BA", "robust - Huber", "robust - Tukey", "robust - Cauchy"])
            plt.grid(True)
            plt.xlabel("Iterations")

            plt.subplot(2, 2, 2)
            plt.title("Inliers evolution\nPoses")
            plt.plot(self.num_inliers_r_ba, linewidth=2)
            plt.plot(self.num_inliers_r_hu, linewidth=2)
            plt.plot(self.num_inliers_r_tu, linewidth=2)
            plt.plot(self.num_inliers_r_cau, linewidth=2)
            plt.legend(["classic BA", "robust - Huber", "robust - Tukey", "robust - Cauchy"])
            plt.grid(True)
            plt.xlabel("Iterations")

            plt.subplot(2, 2, 3)
            plt.title("Chi evolution\nProjections")
            plt.plot(self.chi_stats_p_ba, linewidth=2)
            plt.plot(self.chi_stats_p_hu, linewidth=2)
            plt.plot(self.chi_stats_p_tu, linewidth=2)
            plt.plot(self.chi_stats_p_cau, linewidth=2)
            plt.legend(["classic BA", "robust - Huber", "robust - Tukey", "robust - Cauchy"])
            plt.grid(True)
            plt.xlabel("Iterations")

            plt.subplot(2, 2, 4)
            plt.title("Inliers evolution\nProjections")
            plt.plot(self.num_inliers_p_ba, linewidth=2)
            plt.plot(self.num_inliers_p_hu, linewidth=2)
            plt.plot(self.num_inliers_p_tu, linewidth=2)
            plt.plot(self.num_inliers_p_cau, linewidth=2)
            plt.legend(["classic BA", "robust - Huber", "robust - Tukey", "robust - Cauchy"])
            plt.grid(True)
            plt.xlabel("Iterations")

        plt.style.use('bmh')
        _plot_trajectory()
        _plot_chi_and_inliers()
        plt.show()


def retrieve_method(arg):
    if arg == "CAUCHY":
        return RobustMethod.CAUCHY
    elif arg == "HUBER":
        return RobustMethod.HUBER
    elif arg == "TUKEY":
        return RobustMethod.TUKEY
    else:
        return RobustMethod.NONE


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main")
    # general
    parser.add_argument(
        "-k", "--kind",
        help="A string indicating which kind of run we want to perform: BA (Bundle Adjustment), RBA (Robust Bundle Adjustment) or COMPARISON (BA + every RBA)",
        choices=["BA", "RBA", "COMPARISON"],
        type=str,
        default="BA")

    parser.add_argument(
        "-i", "--iterations",
        help="An integer indicating the number of iteration that BA/RBA must perform",
        type=int,
        default=20)

    parser.add_argument(
        "-d", "--damping",
        help="An float number indicating the damping factor of the BA/RBA",
        type=float,
        default=1.0)

    parser.add_argument(
        "-t", "--threshold",
        help="An float number indicating the inlier kernel threshold for BA/RBA",
        type=float,
        default=1e3)

    parser.add_argument(
        "-o", "--optimize",
        help="A boolean indicating if you want to perform an 'only-landmark' pre optimization, so to enhance the initial guess.",
        type=bool,
        default=False)

    # rba
    parser.add_argument(
        "-m", "--method",
        help="[Only for RBA] A string indicating the specific robustifier to use: CAUCHY, HUBER, TUKEY, and NONE (NONE=>BA)",
        choices=["CAUCHY", "HUBER", "TUKEY", "NONE"],
        type=str,
        default="HUBER")

    parser.add_argument(
        "-p", "--param",
        help="[Only for RBA] A float indicating the specific value for the related robustifier",
        type=float,
        default=1.0)

    # comparison
    args = parser.parse_args()

    print("ARGUMENTS", args)

    pms = PlanarMonocularSLAM(damping=args.damping, kernel_threshold=args.threshold, num_iterations=args.iterations)
    pms.read_data()

    # ===== TRIANGULATION =====
    pms.triangulate()

    if args.kind == "BA":
        if args.optimize:
            pms.pre_optimization()
        # ===== BUNDLE ADJUSTMENT (~1 min) =====
        pms.ba()
        pms.plot()
    elif args.kind == "RBA":
        if args.optimize:
            pms.pre_optimization()
        # ===== ROBUST BUNDLE ADJUSTMENT  (~1 min) =====
        pms.rba(robust_method=retrieve_method(args.method), robust_param=args.param)
        pms.plot()
    elif args.kind == "COMPARISON":
        if args.optimize:
            pms.pre_optimization()
        # ===== BUNDLE ADJUSTMENT + ROBUST BUNDLE ADJUSTMENTS + PLOT  (~4.5 min) =====
        pms.comparison()
    else:
        print("[ERR] Wrong argument for 'kind'")