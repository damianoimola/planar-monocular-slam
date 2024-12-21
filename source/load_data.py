import os

import numpy as np

from source.classes import Trajectory, Measurement, SinglePoint


def load_camera_file():
    path = '../03-PlanarMonocularSLAM/data/camera.dat'
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]

    camera_matrix = np.array([
        lines[1].split(),
        lines[2].split(),
        lines[3].split()
    ], dtype=np.float32)

    camera_transformation = np.array([
        lines[5].split(),
        lines[6].split(),
        lines[7].split(),
        lines[8].split()
    ], dtype=np.float32)

    z_near = lines[9].split()[1]
    z_far = lines[10].split()[1]
    width = lines[11].split()[1]
    height = lines[12].split()[1]

    print("### CAMERA FILE LOADED")
    return camera_matrix, camera_transformation, z_near, z_far, width, height


def load_trajectory_file():
    path = '../03-PlanarMonocularSLAM/data/trajectory.dat'
    with open(path, 'r') as f:
        lines = [line.strip().split() for line in f]

        trajectory = Trajectory()
        for line in lines: trajectory.add(line[0], (line[1:4]), (line[4:7]))
        # trajectory = [[line[0], (line[1:4]), (line[4:7])] for line in lines]
    print("### TRAJECTORY FILE LOADED")
    return trajectory


def load_world_file():
    path = '../03-PlanarMonocularSLAM/data/world.dat'
    with open(path, 'r') as f:
        lines = [line.strip().split() for line in f]
        world_data = [[line[0], (line[1:4])] for line in lines]
    print("### WORLD FILE LOADED")
    return world_data


def load_measurement(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]

    seq = int(lines[0].split()[1])
    gt_pose = [lines[1].split()[1], lines[1].split()[2], lines[1].split()[3]]
    odom_pose = [lines[2].split()[1], lines[2].split()[2], lines[2].split()[3]]

    points = []
    for line in lines[3:]:
        # some files have last line blank
        if line != '':
            split = line.split()
            # [curr_id, actual_id, (point_x, point_y))
            points.append(SinglePoint(split[1], split[2], (split[3], split[4])))
    """
        [
            seq,
            gt_pose,
            odom_pose,
            [
                1 -> [curr_id, actual_id, (point_x, point_y)],
                2 -> [curr_id, actual_id, (point_x, point_y)],
                ...
            ]
        ]
    """
    return Measurement(seq, gt_pose, odom_pose, points)



def load_measurement_files():
    data_folder = '../03-PlanarMonocularSLAM/data'

    measurements_data = []
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f not in ['camera.dat', 'trajectory.dat', 'world.dat']:
                measurements_data.append(load_measurement(os.path.join(root, f)))
    print("### MEASUREMENT FILES LOADED")
    return measurements_data

