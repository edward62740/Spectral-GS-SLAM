<div align="center">

# Spectral GS-SLAM

[Edward Beng Wai Tan](https://edward62740.github.io/), [Siew-Kei Lam](https://siewkeilam.github.io/ei-research-group/index.html), Dongshuo Zhang

### IROS 2026


Please refer to the [website](https://edward62740.github.io/spectralgsslam/) for visualizations or the [paper](https://arxiv.org/abs/2606.21258) for more details.
</div>

## Overview
Real-time (>30FPS) 3DGS SLAM systems typically use ICP or feature matching for tracking. Thus, they inherit the same limitations under structureless and textureless scenes as similar classical SLAM systems.

These scenes, although transiently common indoors (e.g., corridors, white walls), can compromise tracking and lead to unusable systems at deployment.

We use Hessian eigendecomposition techniques to detect and augment the ICP with prior information along deficient axes, and do so via a novel Gaussian-aware technique, which detects the scene's information deficit (covariance) via the existing Gaussian map.

ORB is specifically used as the prior for its complementary strengths compared to ICP and low computational cost. Future adaptations of our work, like using IMU are technically feasible using the same detection mechanism, since it is a property of the scene and not of the prior.

## Setup

The first step of the setup is similar to that of [GS-ICP SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM); if you encounter problems with the C++ compilation, their issues section may have already addressed your issue. The same goes for the gaussian rasterizer.

### Building fast_gicp

The tracker requires the custom `fast_gicp` in `submodules/`, which carries the degeneracy detection and
prior fusion code.

Requires PCL, Eigen3 and OpenMP on the system. Assuming you have installed the dependencies in GS-ICP SLAM,

```bash
conda activate gsicpslam
pip uninstall -y pygicp
pip install --no-build-isolation ./submodules/fast_gicp
```


## Datasets
**TUM** — `bash download_tum.sh`. 


**Replica** — download with `bash download_replica.sh`, then restructure to the layout
(`results/` split into `images/` and `depth_images/`):

```
room0/
    images/          frame000000.jpg ...
    depth_images/    depth000000.png ...
    traj.txt
```

## Run
`DATASET_PATH` points at the directory holding the sequences:

Results are written to `experiments/results` (override with `OUTPUT_PATH`). A single sequence:

```bash
python gs_icp_slam.py \
    --dataset_path /path/to/Replica/room0 \
    --config configs/Replica/caminfo.txt \
    --use_fusion --save_results
```
or simply run the shell scripts (e.g., `tum.sh`) with the relevant sequences uncommented.

`--use_fusion` enables our method.
Add `--rerun_viewer` for live visualization. You may have some issues with the rerun version, in which case those sections can be removed if visualization is not needed.

The detection threshold and Gaussian set selection for the planarity score can be found in the tracker. In general, for larger maps, especially in corridors or maps with perpendicular walls, it's recommended to use the `obs` mode.

## Future work
- [x] Release initial version with prior
- [x] Apply v1.4 fix and src cov scoring
- [ ] Release final version tuned on real-world tests
- [ ] Real-world demo and ToF recordings


## Reference environment

Results in the paper were produced on this configuration:

|||
|---|---|
| GPU / CPU | RTX 4090 (sm_89) / i9-13900KF |
| Driver / CUDA | 580.159.03 / 13.0 (nvcc 13.0.88) |
| PyTorch / cuDNN | 2.11.0+cu130 / 9.19.0 |
| Python / numpy | 3.10.20 / 1.26.4 |
| OpenCV / Open3D | 4.13.0 / 0.19.0 |
| rerun / pybind11 | 0.31.4 / 3.0.4 |


Please note that on `nostructure_far`, stalling near the end of the sequence is an **expected limitation**, due to the lack of ORB features (the details are explained in the paper). You can optionally enable CLAHE in our code.

There is a certain level of stochasticity we found from using slightly different versions of OpenCV, or other dependencies. This is most pronounced in sequences with higher ATE.

## Acknowledgements

This work builds on [GS-ICP SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM), thus please cite their work if you use this implementation. <br>
We also thank:
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[fast_gicp](https://github.com/SMRT-AIST/fast_gicp).

## Cite
```
@misc{tan2026spectralgsslamobservabilityawaredegeneracyrobust,
  title         = {Spectral GS-SLAM: Observability-Aware, Degeneracy-Robust
                   Tracking for Real-Time 3D Gaussian Splatting SLAM},
  author        = {Edward Beng Wai Tan and Siew-Kei Lam and Dongshuo Zhang},
  year          = {2026},
  eprint        = {2606.21258},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.21258}
}
```
