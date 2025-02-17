import numpy as np



def get_camera_pose(XR, cam_pose):
    """get absolute poses for all cameras"""
    X_cam = np.eye(4)
    # 2x2 rotation matrix
    X_cam[:2, :2] = XR[:2, :2]
    # 2D translation components
    X_cam[:2, 3] = XR[:2, 2]
    # camera pose in world coordinates
    return X_cam @ cam_pose


def direction_from_img_coordinates(img_coord, invK):
    """pixel coordinates into 3D ray directions in the camera frame"""
    # convert to homogeneous coordinates (x, y) -> (x, y, 1)
    img_coord = np.append(img_coord, 1)
    # compute the direction vector in camera coordinates
    d = invK @ img_coord
    # normalized vector (unit direction vector)
    return d / np.linalg.norm(d)


def triangulate_multiple_views(points, directions):
    """best estimate of landmark position given multiple camera observations
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 1))

    # looping over ray observations
    for i in range(points.shape[1]):
        a, b, c = directions[:, i]
        x, y, z = points[:, i]

        # building matrix A
        A[0, 0] += 1 - a * a
        A[0, 1] += -a * b
        A[0, 2] += -a * c
        A[1, 1] += 1 - b * b
        A[1, 2] += -b * c
        A[2, 2] += 1 - c * c

        # building vector B
        B[0, 0] += (1 - a * a) * x - a * b * y - a * c * z
        B[1, 0] += -a * b * x + (1 - b * b) * y - b * c * z
        B[2, 0] += -a * c * x - b * c * y + (1 - c * c) * z

    # symmetry adjustments
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]

    # this is the equivalent of minimizing the sum of squared perpendicular distances
    P = np.linalg.solve(A, B)
    return P.flatten()


def triangulate_landmarks(XR_guess, Zp, projection_associations, id_landmarks, cam_pose, invK, num_poses, num_landmarks):
    # deep copy
    new_Zp = Zp.copy()
    new_projection_associations = projection_associations.copy()
    new_projection_associations = new_projection_associations.astype(np.float32)

    # get all camera poses in world coordinates
    X_CAM = np.array([get_camera_pose(XR_guess[:, :, i], cam_pose) for i in range(num_poses)])

    new_id_landmarks = []
    XL_guess = []

    for current_landmark in range(num_landmarks):
        # select only the poses and projections relevant to the current landmark
        idx = np.where(projection_associations[1, :] == id_landmarks[current_landmark])[0]
        poses = projection_associations[0, idx]
        projections = Zp[:, idx]

        # skip landmarks with fewer than 2 observations
        # a single observation cannot be triangulated
        if poses.shape[0] < 2:
            new_Zp[:, idx] = np.nan
            new_projection_associations[:, idx] = np.nan
            continue

        points = []
        directions = []

        for idx_pose in range(len(poses)):
            # extracts the translation vector (position of the camera in world space)
            points.append(X_CAM[poses[idx_pose], :3, 3])
            # extracts the rotation matrix (from camera coordinates to world coordinates)
            R = X_CAM[poses[idx_pose], :3, :3]
            # retrieve projections of a pose
            proj = projections[:, idx_pose]
            # converts 2D image coordinates to a unit vector direction in the camera frame
            direction_vector = direction_from_img_coordinates(proj, invK)
            # rotates the direction vector from camera frame to world frame
            directions.append(R @ direction_vector)
        # uses least-squares triangulation to compute the landmark's 3D coordinates
        landmark_3d = triangulate_multiple_views(np.array(points).T, np.array(directions).T)
        XL_guess.append(landmark_3d)

        # track successfully triangulated landmarks
        new_id_landmarks.append(id_landmarks[current_landmark])
        # update number of landmarks id
        new_projection_associations[1, idx] = len(new_id_landmarks)

    XL_guess = np.array(XL_guess).T
    # keeps only valid columns (columns without 'nan' inside)
    # removing landmarks that were never observed correctly
    valid_idx = ~np.isnan(new_Zp).all(axis=0)
    # only valid landmark observations are kept (only valid columns indices)
    new_Zp = new_Zp[:, valid_idx]
    # only valid projection associations (projections of valid landmarks)
    new_projection_associations = new_projection_associations[:, valid_idx]

    return XL_guess, new_Zp, new_projection_associations, np.array(new_id_landmarks)
