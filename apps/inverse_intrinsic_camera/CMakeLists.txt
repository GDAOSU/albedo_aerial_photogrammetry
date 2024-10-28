GET_FILENAME_COMPONENT(AppName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${AppName} main.cpp)

# Ceres Solver requires CXX11_ABI
# target_compile_definitions(${AppName} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1 -DFIT_INVERSE_COEFFS)
target_link_libraries(${AppName} mysys Eigen3::Eigen)

set_target_properties(${AppName} PROPERTIES INSTALL_RPATH
                                            "${CMAKE_INSTALL_RPATH}")
install(TARGETS ${AppName})