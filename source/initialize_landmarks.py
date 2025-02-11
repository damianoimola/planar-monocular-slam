import numpy as np



def get_camera_pose(XR, cam_pose):
    X_cam = np.eye(4)
    X_cam[:2, :2] = XR[:2, :2]
    X_cam[:2, 3] = XR[:2, 2]
    return X_cam @ cam_pose


def direction_from_img_coordinates(img_coord, invK):
    img_coord = np.append(img_coord, 1)  # Convert to homogeneous coordinates
    d = invK @ img_coord
    return d / np.linalg.norm(d)


def triangulate_multiple_views(points, directions):
    A = np.zeros((3, 3))
    B = np.zeros((3, 1))

    for i in range(points.shape[1]):
        a, b, c = directions[:, i]
        x, y, z = points[:, i]

        A[0, 0] += 1 - a * a
        A[0, 1] += -a * b
        A[0, 2] += -a * c
        A[1, 1] += 1 - b * b
        A[1, 2] += -b * c
        A[2, 2] += 1 - c * c

        B[0, 0] += (1 - a * a) * x - a * b * y - a * c * z
        B[1, 0] += -a * b * x + (1 - b * b) * y - b * c * z
        B[2, 0] += -a * c * x - b * c * y + (1 - c * c) * z

    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]

    P = np.linalg.solve(A, B)
    return P.flatten()


def initialize_landmarks(XR_guess, Zp, projection_associations, id_landmarks, cam_pose, invK, num_poses, num_landmarks):
    # deep copy
    new_Zp = Zp.copy()
    new_projection_associations = projection_associations.copy()
    # TODO CHECK
    new_projection_associations = new_projection_associations.astype(np.float32)

    # get all camera poses in world coordinates
    X_CAM = np.array([get_camera_pose(XR_guess[:, :, i], cam_pose) for i in range(num_poses)])

    new_id_landmarks = []
    XL_guess = []

    for current_landmark in range(num_landmarks):
        # Select only the poses and projections relevant to the current landmark
        idx = np.where(projection_associations[1, :] == id_landmarks[current_landmark])[0]
        poses = projection_associations[0, idx]
        projections = Zp[:, idx]

        if poses.shape[0] < 2:
            new_Zp[:, idx] = np.nan
            new_projection_associations[:, idx] = np.nan
            continue

        points = []
        directions = []

        for current_pose in range(len(poses)):
            points.append(X_CAM[poses[current_pose], :3, 3])
            R = X_CAM[poses[current_pose], :3, :3]
            directions.append(R @ direction_from_img_coordinates(projections[:, current_pose], invK))

        XL_guess.append(triangulate_multiple_views(np.array(points).T, np.array(directions).T))
        new_id_landmarks.append(id_landmarks[current_landmark])
        new_projection_associations[1, idx] = len(new_id_landmarks)

    XL_guess = np.array(XL_guess).T

    valid_idx = ~np.isnan(new_Zp).all(axis=0)
    new_Zp = new_Zp[:, valid_idx]
    new_projection_associations = new_projection_associations[:, valid_idx]

    return XL_guess, new_Zp, new_projection_associations, np.array(new_id_landmarks)
