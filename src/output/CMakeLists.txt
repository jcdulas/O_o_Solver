# Module `output`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_output_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_output STATIC ${oos_output_sources})
add_library(oos::output ALIAS oos_output)
target_include_directories(oos_output PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_output PUBLIC oos::aero)
oos_enable_warnings(oos_output)
