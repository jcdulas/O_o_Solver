# Module `elements`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_elements_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_elements STATIC ${oos_elements_sources})
add_library(oos::elements ALIAS oos_elements)
target_include_directories(oos_elements PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_elements PUBLIC oos::input)
oos_enable_warnings(oos_elements)
