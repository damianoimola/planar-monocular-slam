# Planar Monocular SLAM
Our problem can be identified as a planar (z=0) Multi-point Visual SLAM. 

Landmarks are identifiable with known position.\
Landmarks are observed by a 3D sensor from multiple pose.\
We want to determine:
1. Visual landmark location (2D pixels $\rightarrow \mathbb{R}^2$)
2. Robot pose (3D point + 3D orientation $\rightarrow SE(3)$)
so the overall state is composed by $6\cdot\texttt{poses} + 3\cdot\texttt{landmarks}$, so let's assume that the number of poses will be $N$ while the number of available landmarks is $M$, we obtain $$6N+3M\quad\text{total variables}$$

## State
$$X \in\mathbb{R}^{6N+3M} : X = \lbrace X_r^{[1]}, \dots, X_r^{[N]}, X_l^{[1]},\dots,X_l^{[M]}\rbrace$$
with $$\begin{align}X_r^{[n]}\in SE(3)&: X_r^{[n]}=(R^{[n]}| t^{[n]}) \\ X_l^{[m]}\in\mathbb{R}^3 &: X_l^{[m]}=(x^{[m]}, y^{[m]}, x^{[m]})\end{align}$$


## Perturbation
$$\Delta X \in\mathbb{R}^{6N+3M} : \Delta X = \left( \Delta x_r^{[1]\top}, \dots, \Delta x_r^{[N]\top}, \Delta x_l^{[1]\top},\dots,\Delta x_l^{[M]\top}\right)$$
with $$\begin{align}
\Delta x_r^{[n]\top}\in \mathbb{R}^6&: \Delta x_r^{[n]\top}=\left(\Delta x^{[n]}, \Delta y^{[n]}, \Delta z^{[n]}, \Delta \alpha_x^{[n]}, \Delta \alpha_y^{[n]}, \Delta \alpha_z^{[n]}\right) \\ 
\Delta x_l^{[m]\top}\in\mathbb{R}^3 &: \Delta x_l^{[m]\top}=\left(\Delta x^{[m]}, \Delta y^{[m]}, \Delta z^{[m]}\right)
\end{align}$$


## Box plus
$$\begin{align}
X' &= X\boxplus\Delta x \\
X_r^{[n]'} &= X_r^{[n]}\boxplus\Delta x_r^{[n]} = v2t\left(\Delta x_r^{[n]}\right) X_r^{[n]}\\
X_l^{[m]'} &= X_l^{[m]}+\Delta x_l^{[m]}
\end{align}$$
as you can see, no boxplus operator it's getting applied to the measurements, since they live in Euclidean space (i.e. $\mathbb{R}^3$)


## State observations (meas, preds, errors)
let $\mathbf{z}^{[n, m]}$ be the measurement of the landmark $m$ performed by the robot at pose $n$, so, it will be $$
\mathbf{z}^{[n, m]}\in\mathbb{R}^3 : z^{[n, m]}=\left(x^{[n, m]},y^{[n, m]},z^{[n, m]}\right)$$in this way, we obtain 
$$\begin{align}
h^{[n, m]}(X) & = X_r^{[n]}X_l^{[m]} \\
e^{[n, m]}(X) & = X_r^{[n]}X_l^{[m]} - \mathbf{z}^{[n, m]}\\
e^{[n, m]}(X\boxplus\Delta x) & = v2t\left(\Delta x_r^{[n]}\right) X_r^{[n]}(X_l^{[m]}+\Delta x_l^{[m]}) - \mathbf{z}^{[n, m]}\\
J^{[n, m]} &= \left(\dots \mathbf{0}_{3\times6}\dots J_r^{[n, m]} \dots \mathbf{0}_{3\times6} \mathbf{0}_{3\times3} \dots J_l^{[n, m]}\dots \mathbf{0}_{3\times3}\right)
\end{align}$$


## Pixels observations (meas, preds, errors)
let $\mathbf{z}^{[k]}$ be the measurement of the landmark $m$ performed by the robot at pose $n$, so, it will be $$
\mathbf{z}^{[k]}\in\mathbb{R}^3 : z^{[k]}=\left(u^{[k]}, v^{[k]}\right)$$
in this way, we obtain
 $$\begin{align}
h^{[n]}(\mathbf{X})&=\pi(K\underbrace{\mathbf{X}p_w^{[n]}}_{\displaystyle h_{\text{icp}}^{[n]}(\mathbf{X})})\\[10pt]
\hat{p}^{[n]}_w&=h_{\text{icp}}^{[n]}=\mathbf{X}p_w^{[n]}\\[10pt]
\hat{p}^{[n]}_{cam}&=K\hat{p}^{[n]}_w\\[10pt]
e^{[n,m]}(\mathbf{X})&=h^{[i]}(\mathbf{X})-\mathbf{z}^{[m]}\\
&=\pi(K\mathbf{X}p_w^{[n]})-\mathbf{z}^{[m]}\\[10pt]
e^{[n,m]}(\mathbf{X}\boxplus\Delta\mathbf{x})&=\pi(Kh_{\text{icp}}^{[n]}(\mathbf{X}\boxplus\Delta\mathbf{x}))-\mathbf{z}^{[m]}\\[10pt]
J^{[n]}&=J_{\text{proj}}(\hat{p}^{[n]}_{cam})KJ_{\text{icp}}^{[n]}\\[10pt]
J_{\text{icp}}^{[n]}&=\begin{pmatrix}
I&-\left[\hat{p}^{[n]}\right]_\times
\end{pmatrix}
\end{align}$$
