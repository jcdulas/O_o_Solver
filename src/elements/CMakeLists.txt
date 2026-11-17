# Module `elements`. Sources are found by pattern: a new .cpp needs no edit here.
file(GLOB_RECURSE oos_elements_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(oos_elements STATIC ${oos_elements_sources})
add_library(oos::elements ALIAS oos_elements)
target_include_directories(oos_elements PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_elements PUBLIC oos::input)
oos_enable_warnings(oos_elements)

# The kernels of the mixed hybrid formulation are dense loops over the functions of an element (a shell of degree 4 has
# 88 of them, at 144 points). Under Emscripten they are compiled with the 128-bit vector instructions of WebAssembly,
# which the compiler uses by itself in those loops (no intrinsic: SIMD intrinsics stay in src/linalg/kernels). The
# operations vectorised are element by element, in the order of the scalar loops: the results are the same.
if(EMSCRIPTEN)
    file(GLOB_RECURSE oos_mixed_hybrid_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mixedHybrid/*.cpp")
    set_source_files_properties(${oos_mixed_hybrid_sources} PROPERTIES COMPILE_OPTIONS "-msimd128")
endif()
