get_filename_component(AppName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${AppName} main.cpp TinyRender.cpp)

target_link_libraries(${AppName} ${APP_LIBS} mysys embcommon assimp::assimp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${AppName} OpenMP::OpenMP_CXX)
endif()

set_target_properties(${AppName} PROPERTIES INSTALL_RPATH
                                            "${CMAKE_INSTALL_RPATH}")

install(TARGETS ${AppName})
