# Module `base`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_base_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_base STATIC ${oos_base_sources})
add_library(oos::base ALIAS oos_base)
target_include_directories(oos_base PUBLIC "${PROJECT_SOURCE_DIR}/include")
find_package(Threads REQUIRED)
target_link_libraries(oos_base PUBLIC Threads::Threads)
oos_enable_warnings(oos_base)
