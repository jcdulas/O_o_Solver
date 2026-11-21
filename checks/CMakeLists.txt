# Native checks of the element kernels and of the orderings (OOS_BUILD_NATIVE_CHECKS): a check needs no reader, so it
# runs in the native build. Sources are found by pattern: a new check file needs no edit here.
file(GLOB_RECURSE oos_native_check_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(oos_native_checks ${oos_native_check_sources})
target_include_directories(oos_native_checks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(oos_native_checks PRIVATE oos::api)
oos_enable_warnings(oos_native_checks)
