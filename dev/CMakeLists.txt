add_executable(test_environment_map test_environment_map.cpp)

target_link_libraries(test_environment_map spdlog::spdlog OpenEXR::OpenEXR mysys)
