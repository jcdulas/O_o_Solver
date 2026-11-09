# Module `api`: the simple entry points of the solver (ModelInput, the analyses) that the WebAssembly module exposes to
# JavaScript, in plain C++ so that they also build natively. Sources are found by pattern.
file(GLOB_RECURSE oos_api_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_api STATIC ${oos_api_sources})
add_library(oos::api ALIAS oos_api)
target_include_directories(oos_api PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_api PUBLIC oos::design)
oos_enable_warnings(oos_api)
