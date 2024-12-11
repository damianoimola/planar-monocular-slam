import os

def load_camera_file():
    path = '../03-PlanarMonocularSLAM/data/camera.dat'
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]

    camera_matrix = [
        lines[1].split(),
        lines[2].split(),
        lines[3].split()
    ]

    camera_transformation = [
        lines[5].split(),
        lines[6].split(),
        lines[7].split(),
        lines[8].split()
    ]

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
        trajectory = [[line[0], (line[1:4]), (line[4:7])] for line in lines]
    print("### TRAJECTORY FILE LOADED")
    return trajectory


def load_world_file():
    path = '../03-PlanarMonocularSLAM/data/world.dat'
    with open(path, 'r') as f:
        lines = [line.strip().split() for line in f]
        world_data = [[line[0], (line[1:4])] for line in lines]
    print("### WORLD FILE LOADED")
    return world_data


def load_measurement_file():
    data_folder = '../03-PlanarMonocularSLAM/data'

    measurements_data = []
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f not in ['camera.dat', 'trajectory.dat', 'world.dat']:
                measurements_data.append(...)
    return measurements_data

