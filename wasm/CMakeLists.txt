# The WebAssembly module (64-bit WebAssembly, Memory64: see the top CMakeLists.txt): the solver compiled with Emscripten,
# its embind interface (src/Bindings.cpp), and the JavaScript wrapper (js/). The package is assembled in <build>/dist:
#   oos_solver_core.js / .wasm   the module made by Emscripten
#   index.js, index.d.ts         the wrapper a web application imports
#   package.json
add_executable(oos_solver_core src/Bindings.cpp src/OptimizationBindings.cpp)
target_link_libraries(oos_solver_core PRIVATE oos::api)
set_target_properties(oos_solver_core PROPERTIES
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist")
target_link_options(oos_solver_core PRIVATE
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createOosSolverCore
    -sENVIRONMENT=web,worker,node
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=16GB
    -sINITIAL_MEMORY=64MB
    # The solver recurses (the elimination tree of the sparse factor, the nested dissection): a larger stack than the
    # default 64 KB.
    -sSTACK_SIZE=8MB
    -sFORCE_FILESYSTEM=1
    "-sEXPORTED_RUNTIME_METHODS=['FS']")

function(oos_wasm_workers target)
    if(OOS_WASM_THREADS)
        math(EXPR worker_count "${OOS_WASM_MAX_THREADS} - 1")
        target_link_options(${target} PRIVATE
            "-sPTHREAD_POOL_SIZE=${worker_count}"
            -sPTHREAD_POOL_SIZE_STRICT=2
            -sDEFAULT_PTHREAD_STACK_SIZE=8MB
            -sALLOW_BLOCKING_ON_MAIN_THREAD=0)
    endif()
endfunction()
oos_wasm_workers(oos_solver_core)

if(OOS_WASM_THREADS)
    set(OOS_THREADED_JS true)
    set(OOS_MAX_THREADS_JS ${OOS_WASM_MAX_THREADS})
else()
    set(OOS_THREADED_JS false)
    set(OOS_MAX_THREADS_JS 1)
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/dist")
configure_file(js/build-info.js.in "${CMAKE_BINARY_DIR}/dist/build-info.js" @ONLY)

if(OOS_BUILD_WASM_TESTS)
    add_executable(oos_threading_test test/ThreadPoolChecks.cpp)
    target_link_libraries(oos_threading_test PRIVATE oos::api)
    set_target_properties(oos_threading_test PROPERTIES SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist")
    target_link_options(oos_threading_test PRIVATE -lembind -sMODULARIZE=1 -sEXPORT_ES6=1
        -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=16GB
        -sINITIAL_MEMORY=64MB -sSTACK_SIZE=8MB)
    oos_wasm_workers(oos_threading_test)
endif()

# The wrapper, its types and the package description, next to the module.
foreach(file index.js index.d.ts optimization.js optimization.d.ts)
    set_property(TARGET oos_solver_core APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/js/${file}")
    add_custom_command(TARGET oos_solver_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/js/${file}"
                "${CMAKE_BINARY_DIR}/dist/${file}")
endforeach()
add_custom_command(TARGET oos_solver_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/package.json"
            "${CMAKE_BINARY_DIR}/dist/package.json")
