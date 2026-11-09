# Module `solutions`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_solutions_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_solutions STATIC ${oos_solutions_sources})
add_library(oos::solutions ALIAS oos_solutions)
target_include_directories(oos_solutions PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_solutions PUBLIC oos::output)
oos_enable_warnings(oos_solutions)
