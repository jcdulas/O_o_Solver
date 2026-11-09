# Module `aero`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_aero_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_aero STATIC ${oos_aero_sources})
add_library(oos::aero ALIAS oos_aero)
target_include_directories(oos_aero PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_aero PUBLIC oos::reduction)
oos_enable_warnings(oos_aero)
