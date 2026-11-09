# Module `design`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_design_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_design STATIC ${oos_design_sources})
add_library(oos::design ALIAS oos_design)
target_include_directories(oos_design PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_design PUBLIC oos::solutions)
oos_enable_warnings(oos_design)
