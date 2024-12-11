# planar-monocular-slam

Differential Drive equipped with a monocular camera
Input
- Integrated dead reckoning (wheeled odometry)
- Stream of point projections with “id”
- Camera Parameters
- Extrinsics (pose of camera on robot)
- Intrinsics (K)
Output
- Trajectory (estimate vs gt)
- 3D points (estimate vs gt)
- Error values (rotation and translation)
How
- Bootstrap the system by triangulating the initial set of points with the odometry guess
- Bundle Adjustment (total least squares) at the end
