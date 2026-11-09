# Module `linalg`. Sources are found by pattern: a new .cpp needs no edit here.
# The kernel families (kernels/reference, kernels/avx2, kernels/avx512) are compiled apart, each with its own target
# flags (cmake/OosKernels.cmake); everything else at the baseline x86-64.
set(oos_all_kernel_families reference avx2 avx512)
if(OOS_IS_WASM)
    # WebAssembly has no x86 vector units: only the reference family is compiled.
    set(oos_kernel_families reference)
else()
    set(oos_kernel_families reference avx2 avx512)
endif()

file(GLOB_RECURSE oos_linalg_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(family IN LISTS oos_all_kernel_families)
    list(FILTER oos_linalg_sources EXCLUDE REGEX "/kernels/${family}/")
endforeach()

add_library(oos_linalg STATIC ${oos_linalg_sources})
add_library(oos::linalg ALIAS oos_linalg)
target_include_directories(oos_linalg PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oos_linalg PUBLIC oos::base)
oos_enable_warnings(oos_linalg)

foreach(family IN LISTS oos_kernel_families)
    file(GLOB_RECURSE family_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${family}/*.cpp")
    oos_add_kernel_family(oos_linalg ${family} ${family_sources})
endforeach()

# The baseline part of the dense kernels (W = L·D, the 2 × 2 pivot formula, the diagonal solve) must round each product
# as the kernel families do: no contraction into fused multiply-adds, even with a higher OOS_BASELINE_ARCH.
target_compile_options(oos_linalg PRIVATE -ffp-contract=off)
