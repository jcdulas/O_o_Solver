cmake_minimum_required(VERSION 3.25)

project(oos VERSION 0.2.6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(OOS_BUILD_WASM "Build the WebAssembly module (needs the Emscripten toolchain)" OFF)
option(OOS_WASM_THREADS "Enable shared-memory workers in the WebAssembly build" OFF)
set(OOS_WASM_MAX_THREADS 8 CACHE STRING "Maximum threads per WebAssembly instance, including the calling thread")
option(OOS_BUILD_WASM_TESTS "Build WebAssembly threading checks" OFF)
option(OOS_BUILD_NATIVE_CHECKS "Build the native checks of the element kernels (checks/)" OFF)
option(OOS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
# Keep the invariant checks (OOS_ASSERT) in optimised builds: a broken invariant stops the run with a message rather
# than giving wrong numbers.
option(OOS_ASSERTIONS "Keep the invariant checks (OOS_ASSERT) in optimised builds" ON)

# WebAssembly uses the reference kernels. Shared-memory workers are enabled by OOS_WASM_THREADS.
#
# The WebAssembly build is 64-bit (Memory64, "wasm64"), like the O_o_Model module it takes its tables from: pointers and
# sizes are 64-bit, so a model may use more than 4 GB of memory. C++ exceptions are the native WebAssembly ones: they
# are how the solver reports errors.
if(EMSCRIPTEN)
    set(OOS_IS_WASM ON)
    add_compile_definitions(OOS_WASM=1)
    add_compile_options(-sMEMORY64=1 -fwasm-exceptions)
    add_link_options(-sMEMORY64=1 -fwasm-exceptions)
    if(OOS_WASM_THREADS)
        if(NOT OOS_WASM_MAX_THREADS MATCHES "^[1-9][0-9]*$" OR OOS_WASM_MAX_THREADS GREATER 64)
            message(FATAL_ERROR "OOS_WASM_MAX_THREADS must be an integer from 1 to 64")
        endif()
        add_compile_options(-pthread)
        add_link_options(-pthread)
        add_compile_definitions(OOS_WASM_MAX_THREADS=${OOS_WASM_MAX_THREADS})
    endif()
else()
    set(OOS_IS_WASM OFF)
endif()

include(OosWarnings)
include(OosKernels)

if(OOS_ASSERTIONS)
    add_compile_definitions(OOS_ASSERTIONS=1)
endif()

if(MSVC)
    add_compile_options(/utf-8 /bigobj)
    add_compile_definitions(NOMINMAX)
endif()

option(OOS_WITH_MODEL "Build the native modelFromDeck bridge" OFF)
if(OOS_WITH_MODEL AND NOT TARGET oor::core)
    find_package(oo-model 0.2.1 CONFIG REQUIRED)
endif()
add_subdirectory(src)
if(OOS_WITH_MODEL)
    target_sources(oos_api PRIVATE native/ModelFromDeck.cpp)
    target_link_libraries(oos_api PUBLIC oor::core)
endif()

option(OOS_BUILD_PYTHON "Build the native CPython extension" OFF)
if(OOS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(OOS_BUILD_NATIVE_CHECKS AND NOT EMSCRIPTEN)
    add_subdirectory(checks)
endif()

if(OOS_BUILD_WASM)
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "OOS_BUILD_WASM needs the Emscripten toolchain (use the 'wasm' preset)")
    endif()
    add_subdirectory(wasm)
endif()

if(PROJECT_IS_TOP_LEVEL AND NOT EMSCRIPTEN AND NOT OOS_BUILD_PYTHON)
    include(cmake/InstallSdk.cmake)
endif()
