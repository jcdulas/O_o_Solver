# Target flags of the dense kernel families (src/linalg/kernels).
#
# The whole project is compiled for the baseline x86-64 (OOS_BASELINE_ARCH), so that one binary runs on every
# x86-64 processor. Only the kernel families are compiled for more: each folder src/linalg/kernels/<family>/ is an object
# library with its own flags, and the family is chosen at run time by CPU detection (linalg/dense/KernelSelection).
# A file of a SIMD family must not define a function with external linkage other than its entry points, nor use an
# inline function shared with baseline code (see include/oos/linalg/kernels/MatrixProduct.hpp).

set(OOS_BASELINE_ARCH "x86-64" CACHE STRING "The -march of everything outside the SIMD kernel families")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|amd64")
    # MSYS2's GCC defaults to -march=nocona; the baseline is stated rather than inherited from the toolchain.
    add_compile_options(-march=${OOS_BASELINE_ARCH} -mtune=generic)
endif()

# The flags of each family. Every family forbids the contraction of a*b+c into a fused multiply-add: the reference
# family must give the same bits on every machine (it is the oracle), whatever the -march; in the SIMD families the only
# fused operations are the explicit ones (intrinsics, __builtin_fma), so that their scalar and vector code give the same
# bits and AVX2 and AVX-512 agree. GCC lowers _mm256_mul_pd and
# _mm256_sub_pd to generic vector arithmetic, which it would otherwise be free to fuse.
set(OOS_KERNEL_FLAGS_reference -ffp-contract=off)
set(OOS_KERNEL_FLAGS_avx2 -mavx2 -mfma -ffp-contract=off)
set(OOS_KERNEL_FLAGS_avx512 -mavx512f -mavx512dq -mavx512vl -mavx2 -mfma -ffp-contract=off)

# oos_add_kernel_family(<target> <family> <sources...>)
# Compiles the sources of one kernel family into an object library and adds its objects to <target>.
function(oos_add_kernel_family target family)
    set(sources ${ARGN})
    if(NOT sources)
        return()
    endif()
    set(objects ${target}_kernels_${family})
    add_library(${objects} OBJECT ${sources})
    target_include_directories(${objects} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_options(${objects} PRIVATE ${OOS_KERNEL_FLAGS_${family}})
    if(MINGW AND NOT family STREQUAL "reference")
        # Windows x64 aligns the stack to 16 bytes only, and GCC spills 32- and 64-byte registers with aligned moves
        # that fault on it (GCC bug 54412). The assembler is told to emit unaligned moves instead (binutils >= 2.38).
        target_compile_options(${objects} PRIVATE -Wa,-muse-unaligned-vector-move)
    endif()
    oos_enable_warnings(${objects})
    target_sources(${target} PRIVATE $<TARGET_OBJECTS:${objects}>)
endfunction()
