#!/bin/bash
conda create -n aerial_albedo python=3.12 -y
conda activate aerial_albedo
mamba install zlib embree3 nlohmann_json assimp openexr doctest ceres-solver eigen spdlog cxxopts -c conda-forge -y
mamba install openimageio scikit-image tqdm imageio trimesh pydensecrf scipy matplotlib seaborn -c conda-forge -y