add_subdirectory(embcommon)

add_library(mysys STATIC CameraModel.cpp utils/sphere_tessellation.cpp
                         utils/writer.cpp utils/AOProps.cpp)
target_link_libraries(
  mysys
  PUBLIC Eigen3::Eigen
  PRIVATE Ceres::ceres nlohmann_json::nlohmann_json OpenEXR::OpenEXR)
target_compile_definitions(
  mysys PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1 -DFIT_INVERSE_COEFFS=1
               -DUSE_CERES_SOLVER=1)
