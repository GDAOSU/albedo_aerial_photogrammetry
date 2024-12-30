#!/bin/bash
conda create -n aerial_albedo python=3.12 -y
conda activate aerial_albedo
conda install zlib tbb-devel embree3 nlohmann_json assimp openexr doctest ceres-solver eigen spdlog cxxopts -c conda-forge -y
conda install opencv shapely openimageio py-openimageio scikit-image tqdm imageio trimesh pydensecrf scipy matplotlib seaborn pymap3d lxml pyproj -c conda-forge -y