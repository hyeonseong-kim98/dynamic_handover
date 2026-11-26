# Robot Control Node with C++

## Installation

1. Install Eigen3:

   ```bash
   sudo apt-get install libeigen3-dev
   ```

2. Install ur_rtde:

   ```bash
   sudo add-apt-repository ppa:sdurobotics/ur-rtde
   sudo apt-get update
   sudo apt install librtde librtde-dev
   ```

3. Install libmodbus:

   ```bash
   sudo apt install libmodbus-dev
   ```

## Load dataset

The public dataset is included as a Git submodule at `run_ws/data/dataset`.
After cloning this repository, initialize it from the repository root:

```bash
git submodule update --init --recursive
```

From this repository's root, run either estimator in your ROS 2 / ZED / MuJoCo environment:

```bash
python run_ws/estimator_optimized.py
python run_ws/estimator_unoptimized.py
```

Motion data is loaded from the dataset submodule.
Keep the local `run_ws/data/assets` robot models and
`run_ws/data/camera_pose/camera_transformation.json` camera calibration for the
robot pipeline. The camera calibration must match the physical setup.

## Python SPGP modules

`run_ws/SparseIAAL/sparse_IAAL.py` provides prediction without temporal ensembling
and preserves the original constructor's positional arguments. It shares the
maintained SPGP implementation with `sparse_IAAL_te.py`, including public dataset
loading, filename-validated caches, numerical safeguards, and observation resets.

The default flow model is fitted separately for each demonstration.
`use_gpr_whole=True` fits one pooled flow and gives every demonstration the same
flow score while retaining them as giver-trajectory candidates. Kernel caching
applies to the default per-demonstration mode.

## Tests

From the repository root, install the test dependencies and run the regression suite:

```bash
python -m pip install numpy scipy
python tests/test_spgp.py -v
```

The tests create small synthetic motion datasets in temporary directories and
require no ROS, ZED camera, MuJoCo, robot hardware, or dataset submodule checkout.
They cover dataset loading, cache validation, stationary receivers, observation
resets, both SPGP modules, forecasting boundaries, and pooled flow prediction.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{kim2025learning,
  author    = {Kim, Hyeonseong and Kim, Chanwoo and Pan, Matthew and Lee, Kyungjae and Choi, Sungjoon},
  title     = {Learning-Based Dynamic Robot-to-Human Handover},
  booktitle = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025},
  pages     = {16319--16325},
  publisher = {IEEE},
  doi       = {10.1109/ICRA55743.2025.11128251},
  url       = {https://ieeexplore.ieee.org/document/11128251/}
}
```
