get_filename_component(AppName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${AppName} main.cpp AmbientOccBaker.cpp)

target_link_libraries(${AppName} ${APP_LIBS} mysys embcommon assimp::assimp)

set_target_properties(${AppName} PROPERTIES INSTALL_RPATH
                                            "${CMAKE_INSTALL_RPATH}")
install(TARGETS ${AppName})
