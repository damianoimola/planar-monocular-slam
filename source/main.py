import numpy as np
import cv2
from scipy.optimize import least_squares

from load_data import load_camera_file, load_trajectory_file, load_world_file, load_measurement_files

class PlanarMonocularSLAM:
    def __init__(self):
        # ===== DATA LOADING =====
        camera_matrix, camera_transformation, z_near, z_far, width, height = load_camera_file()
        trajectory = load_trajectory_file()
        world_data = load_world_file()
        measurement_data = load_measurement_files()

    def triangulation(self, pose1, pose2, points1, points2):
        return



PlanarMonocularSLAM()