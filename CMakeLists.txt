cmake_minimum_required(VERSION 3.25)

project(oos VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(OOS_BUILD_WASM "Build the WebAssembly module (needs the Emscripten toolchain)" OFF)
option(OOS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
# Keep the invariant checks (OOS_ASSERT) in optimised builds: a broken invariant stops the run with a message rather
# than giving wrong numbers.
option(OOS_ASSERTIONS "Keep the invariant checks (OOS_ASSERT) in optimised builds" ON)

# WebAssembly runs on one thread and has no x86 vector units: the build leaves out the worker threads and the AVX2 and
# AVX-512 kernel families (the reference family is used), see cmake/OosKernels.cmake and base/threading/ThreadPool.
#
# The WebAssembly build is 64-bit (Memory64, "wasm64"), like the O_o_Reader module it takes its tables from: pointers and
# sizes are 64-bit, so a model may use more than 4 GB of memory. C++ exceptions are the native WebAssembly ones: they
# are how the solver reports errors.
if(EMSCRIPTEN)
    set(OOS_IS_WASM ON)
    add_compile_definitions(OOS_WASM=1)
    add_compile_options(-sMEMORY64=1 -fwasm-exceptions)
    add_link_options(-sMEMORY64=1 -fwasm-exceptions)
else()
    set(OOS_IS_WASM OFF)
endif()

include(OosWarnings)
include(OosKernels)

if(OOS_ASSERTIONS)
    add_compile_definitions(OOS_ASSERTIONS=1)
endif()

add_subdirectory(src)

if(OOS_BUILD_WASM)
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "OOS_BUILD_WASM needs the Emscripten toolchain (use the 'wasm' preset)")
    endif()
    add_subdirectory(wasm)
endif()
