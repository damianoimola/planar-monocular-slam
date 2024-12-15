# Problem analysis

## Quick recap
- Camera matrix: also called Intrinsic matrix, maps 3D points in the camera coordinated to 2D points in the image coordinates $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
- Essential matrix capture relative rotation and translation between two views $$E=[R|t]$$
- Projection matrix: combines the camera intrinsic and extrinsic parameters $$P = K[R|t]\in\mathbb{R}^{3\times 4}$$ with
  - $R\in\mathbb{R}^{3\times 3}$ = orientation of camera in world
  - $t\in\mathbb{R}^{3\times 1}$ = translation of the camera relative to the world


## World to Image
1) Given:
   1) 3D point in world coordinate $P_{world}=[X, Y, Z, 1]^\top$
   2) Extrinsic matrix: $E$
   3) Intrinsic matrix (camera matrix): $K$
2) WORLD $\rightarrow$ CAMERA Transformation $$P_{camera} = E\cdot P_{world} = [R|t]\cdot P_{world} = R * P_{world}+t$$ where $P_{camera}=[x_c, y_c, z_c]^\top\in\mathbb{R}^3$
3) CAMERA $\rightarrow$ IMAGE Transformation $$P_{image} = K\cdot P_{camera}$$  where $P_{image}=[x, y, z]^\top\in\mathbb{R}^3$
4) HOMOGENEOUS $\rightarrow$ NON-HOMOGENEOUS $$(u,v)= \left(\frac{x}{z}, \frac{y}{z} \right) = \Pi(P_{image})$$

The complete pipeline is the following $$(u,v) = \Pi(K\cdot E\cdot P_{world})$$

## Triangulation

