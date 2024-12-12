# Problem analysis

## World to Image
1) Given:
   1) 3D point in world coordinate $P_{world}=[X, Y, Z, 1]^\top$
   2) Extrinsic matrix: $E$
   3) Intrinsic matrix (camera matrix): $K$
2) WORLD $\rightarrow$ CAMERA Transformation $$P_{camera} = E\cdot P_{world} = [R|t]\cdot P_{world} = R * P_{world}+t$$ where $P_{camera}=[x_c, y_c, z_c]^\top\in\mathbb{R}^3$
3) CAMERA $\rightarrow$ IMAGE Transformation $$P_{image} = K\cdot P_{camera}$$  where $P_{image}=[x, y, z]^\top\in\mathbb{R}^3$
4) HOMOGENEOUS $\rightarrow$ NON-HOMOGENEOUS $$(u,v)= \left(\frac{x}{z}, \frac{y}{z} \right) = \Pi(P_{image})$$

The complete pipeline is the following $$(u,v) = \Pi(K\cdot E\cdot P_{world})$$ or, if thereference system of the world and the camere are not the same, we can compute $(u,v)$ as follows $$(u,v) = \Pi(K\cdot E\cdot P_{world})$$
