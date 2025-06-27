#!/bin/bash
conda create -n aerial_albedo python=3.12 -y
conda activate aerial_albedo
conda install zlib tbb-devel embree3 nlohmann_json assimp openexr doctest ceres-solver eigen spdlog cxxopts -c conda-forge -y
conda install opencv shapely scikit-image tqdm imageio trimesh pydensecrf scipy matplotlib seaborn pymap3d lxml pyproj -c conda-forge -y
#for linuux
conda install openimageio py-openimageio 
#for windows
cd {OIIO_ROOT}
git clone https://github.com/AcademySoftwareFoundation/OpenImageIO .
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX% -DPYTHON=ON -DPYTHON_EXECUTABLE=%CONDA_PREFIX%\python.exe -DUSE_EXTERNAL_PYBIND11=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH=%CONDA_PREFIX% -DZLIB_ROOT=%CONDA_PREFIX%\library -DPNG_ROOT=%CONDA_PREFIX%\library -DFreetype_ROOT=%CONDA_PREFIX%\library -DJXL_ROOT=%CONDA_PREFIX%\library -DOpenImageIO_BUILD_MISSING_DEPS=all 
#for python
