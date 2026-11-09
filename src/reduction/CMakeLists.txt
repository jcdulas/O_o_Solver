# Module `reduction`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_reduction_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_reduction STATIC ${oos_reduction_sources})
add_library(oos::reduction ALIAS oos_reduction)
target_include_directories(oos_reduction PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_reduction PUBLIC oos::assembly)
oos_enable_warnings(oos_reduction)
