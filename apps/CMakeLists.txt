set(APP_LIBS spdlog::spdlog cxxopts::cxxopts nlohmann_json::nlohmann_json)

add_subdirectory(inverse_intrinsic_camera)
add_subdirectory(tinyrender)
add_subdirectory(ambientocc_baker)
