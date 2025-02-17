# 📷 Planar Monocular SLAM

> IMPORTANT: the complete and detailed report is available inside the folder `/report`

## 📌 Project Overview
This project implements a **planar monocular Simultaneous Localization and Mapping (SLAM) system** for a differential drive robot equipped with a single camera. The system integrates:
- 📍 **Wheeled odometry**
- 🎯 **Point projections** for trajectory estimation
- 🗺️ **Landmark mapping**
- 🔧 **Robust Bundle Adjustment** using **Huber, Cauchy, and Tukey M-estimators** to handle outliers effectively.

## 🔬 Features
- **Monocular SLAM** using a single camera
- **Odometry integration** for improved accuracy
- **Triangulation** for 3D point estimation
- **Bundle Adjustment (BA)** for refining trajectory and landmark positions
- **Robust Optimization** using M-estimators:
  - Huber (smooth transition between quadratic and linear loss)
  - Cauchy (fast suppression of large errors)
  - Tukey (aggressive outlier down-weighting)

## 🏗️ Installation
### 🔧 Prerequisites
```bash
conda create --name myenv --file packageslist.txt
```

## 🚀 Usage
### 🏁 Running the SLAM System
Run the main script with the appropriate arguments:
```bash
python main.py --kind BA --iterations 20 --damping 1.0 --threshold 1e3 --optimize False --method HUBER --param 1.0
```
#### Available Options:
- `--kind` : Type of run (`BA`, `RBA`, `COMPARISON`)
- `--iterations` : Number of iterations for BA/RBA
- `--damping` : Damping factor for BA/RBA
- `--threshold` : Inlier kernel threshold for BA/RBA
- `--optimize` : Boolean flag for performing an "only-landmark" pre-optimization
- `--method` : Robustifier method (`CAUCHY`, `HUBER`, `TUKEY`, `NONE`)
- `--param` : Specific value for the chosen robustifier

### ⚙️ Configuration
Modify parameters inside the code or via bash for tuning the system behavior.

## 📊 Results
The system has been tested under two configurations:
1. **Without pre-optimization**: Directly applying Bundle Adjustment (BA/RBA) on the full system.
2. **With pre-optimization**: Refining landmark estimates first, then running BA/RBA.

### 🛠️ Key Findings:
- **Robust BA** improves accuracy and resilience to noisy measurements.
- **Pre-optimization** further enhances mapping precision.
- **Tuning M-estimator parameters** significantly impacts system performance.

## 📝 References
- [Optimal Ray Intersection for Computing 3D Points](https://www.eecs.qmul.ac.uk/~gslabaugh/publications/opray.pdf)
- [GitLab Repository](https://gitlab.com/grisetti/probabilistic_robotics_2024_25/-/tree/main/source/octave/26_total_least_squares)

## 📜 License
This project is licensed under **Creative Commons Attribution 4.0 International**. See [LICENSE](LICENSE) for details.

---
🤖 Happy Mapping! 🎉

