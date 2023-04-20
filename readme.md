# PIP

Code for our CVPR 2022 [paper](https://arxiv.org/abs/2203.08528) "Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors". This repository contains the system implementation and evaluation.  See [Project Page](https://xinyu-yi.github.io/PIP/).

![1](data/figures/1.jpg)

## Usage

### Install dependencies

We use `python 3.7.6`. You should install the newest `pytorch chumpy vctoolkit open3d pybullet qpsolvers cvxopt`.

You also need to compile and install [rbdl](https://github.com/rbdl/rbdl) with python bindings. Also install the urdf reader addon. This library is easy to compile on Linux. For Windows, you need to rewrite some codes and the CMakeLists. We have only tested our system on Windows.

*If the newest `vctoolkit` reports errors, please use `vctoolkit==0.1.5.39`.*

*Installing `pytorch` with CUDA is recommended but not mandatory. During evaluation, the motion prediction can run at ~120fps on CPU, but computing the errors may be very slow without CUDA.*

*If you have configured [TransPose](https://github.com/Xinyu-Yi/TransPose/), just use its environment and install the missing packages including the `rbdl`.*

### Prepare SMPL body model

1. Download SMPL model from [here](https://smpl.is.tue.mpg.de/). You should click `SMPL for Python` and download the `version 1.0.0 for Python 2.7 (10 shape PCs)`. Then unzip it.
2. In `config.py`, set `paths.smpl_file` to the model path.

*If you have configured [TransPose](https://github.com/Xinyu-Yi/TransPose/), just copy its settings here.*

### Prepare physics body model

1. Download the physics body model from [here](https://xinyu-yi.github.io/PIP/files/urdfmodels.zip) and unzip it.
2. In `config.py`, set `paths.physics_model_file` to the body model path.
3. In `config.py`, set `paths.plane_file`  to `plane.urdf`. Please put `plane.obj` next to it.

*The physics model and the ground plane are modified from [physcap](https://github.com/soshishimada/PhysCap_demo_release).*

### Prepare pre-trained network weights

1. Download weights from [here](https://xinyu-yi.github.io/PIP/files/weights.pt).
2. In `config.py`, set `paths.weights_file` to the weights path.

### Prepare test datasets

1. Download DIP-IMU dataset from [here](https://dip.is.tue.mpg.de/). We use the raw (unnormalized) data.
2. Download TotalCapture dataset from [here](https://cvssp.org/data/totalcapture/). You need to download `the real world position and orientation` under `Vicon Groundtruth` in the website and unzip them. The ground-truth SMPL poses used in our evaluation are provided by the DIP authors. So you may also need to contact the DIP authors for them.
3. In `config.py`, set `paths.raw_dipimu_dir` to the DIP-IMU dataset path; set `paths.raw_totalcapture_dip_dir` to the TotalCapture SMPL poses (from DIP authors) path; and set `paths.raw_totalcapture_official_dir` to the TotalCapture official `gt` path. Please refer to the comments in the codes for more details.

*If you have configured [TransPose](https://github.com/Xinyu-Yi/TransPose/), just copy its settings here. **Remember**: you need to rerun the `preprocess.py` as the preprocessing of TotalCapture dataset has been changed to remove the acceleration bias.*

### Run the evaluation

You should preprocess the datasets before evaluation:

```
python preprocess.py
python evaluate.py
```

The pose/translation evaluation results for DIP-IMU and TotalCapture test datasets will be printed/drawn.

### Live Demo

The live demo codes are on the `livedemo` branch. Please checkout this branch.

### About the codes

The authors are too busy to clean up/rewrite the codes. Here are some useful tips:

- In `dynamics.py`, there are many disabled options for the physics optimization. You can try different combinations of the energy terms by enabling the corresponding terms. 

- In Line ~44 in `net.py`:

  ```python
  self.dynamics_optimizer = PhysicsOptimizer(debug=False)
  ```

  set `debug=True` to visualize the estimated motions using pybullet. You may need to clean the cached results and rerun the `evaluate.py`. (e.g., set `flush_cache=True` in `evaluate()` and rerun.)

- In Line ~244 in `dynamics.py`:

  ```python
  if False:   # visualize GRF (no smoothing)
      p.removeAllUserDebugItems()
      for point, force in zip(collision_points, GRF.reshape(-1, 3)):
          p.addUserDebugLine(point, point + force * 1e-2, [1, 0, 0])
  ```

  Enabling this to visualize the ground reaction force. (You also need to set `debug=True` as stated above.) Note that rendering the force lines can be very slow in pybullet. 

- The hyperparameters for the physics optimization are all in `physics_parameters.json`.  If you set `debug=True`, you can adjust these parameters interactively in the pybullet window.

## Citation

If you find the project helpful, please consider citing us:

```
@InProceedings{PIPCVPR2022,
  author = {Yi, Xinyu and Zhou, Yuxiao and Habermann, Marc and Shimada, Soshi and Golyanik, Vladislav and Theobalt, Christian and Xu, Feng},
  title = {Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2022}
}
```

