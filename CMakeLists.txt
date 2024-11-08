cmake_minimum_required(VERSION 3.15)

enable_language(C CXX)

project(aerial_albedo_photogrammetry)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(BUILD_NVVK_APPS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Enable fPIC for pybind11

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

include(GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Default build type: Release" FORCE)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_APPS "Build applications" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ADD_CONDA_TO_RPATH "Add conda to rpath" ON)

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PATH $ENV{CONDA_PREFIX})
  message(STATUS "ADD_CONDA_TO_RPATH: ${ADD_CONDA_TO_RPATH}")
  message(NOTICE "[CONDA] Environment detected: ${CONDA_PATH}")

  # If CMAKE_INSTALL_PREFIX is not set, set it to CONDA_PATH
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CONDA_PATH})
    message(NOTICE "[CONDA] Setting CMAKE_INSTALL_PREFIX to ${CONDA_PATH}")
  elseif(ADD_CONDA_TO_RPATH)
    # Add conda to rpath
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CONDA_PATH}/lib")
    message(NOTICE
            "[CONDA] Setting CMAKE_INSTALL_RPATH to ${CMAKE_INSTALL_RPATH}")
  endif()

  set(CMAKE_PREFIX_PATH
      "${CONDA_PATH};${CONDA_PATH}/lib/cmake;${CMAKE_PREFIX_PATH}")
  message(NOTICE "[CONDA] Setting CMAKE_PREFIX_PATH to ${CMAKE_PREFIX_PATH}")

endif()

find_package(OpenMP REQUIRED)

find_package(cxxopts CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

find_package(embree CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

include_directories(${OpenEXR_INCLUDE_DIRS})
if(DOCTEST_FOUND)
  enable_testing()
endif()

include_directories(src_cpp)
add_subdirectory(src_cpp)
#  add_subdirectory(tests)
add_subdirectory(apps)
add_subdirectory(dev)
