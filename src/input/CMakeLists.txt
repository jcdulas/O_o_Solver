# Module `input`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_input_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_input STATIC ${oos_input_sources})
add_library(oos::input ALIAS oos_input)
target_include_directories(oos_input PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_input PUBLIC oos::linalg)
oos_enable_warnings(oos_input)
