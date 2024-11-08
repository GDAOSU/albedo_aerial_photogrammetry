add_subdirectory(embcommon)

add_library(mysys STATIC CameraModel.cpp
  EnvironmentMap.cpp
  utils/sphere_tessellation.cpp
  utils/writer.cpp utils/AOProps.cpp)
target_link_libraries(
  mysys
  PUBLIC Eigen3::Eigen OpenEXR::OpenEXR
  PRIVATE Ceres::ceres nlohmann_json::nlohmann_json)

target_compile_definitions(mysys PUBLIC -DFIT_INVERSE_COEFFS=1
  -DUSE_CERES_SOLVER=1)
