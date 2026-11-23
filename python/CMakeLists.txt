find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_hint
                OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_hint}")
pybind11_add_module(_native MODULE NO_EXTRAS bindings.cpp)
target_link_libraries(_native PRIVATE oos_api)
install(TARGETS _native LIBRARY DESTINATION oo_solver RUNTIME DESTINATION oo_solver)

# Wheel repair bundles native runtime libraries. Ship their distributor's notices with the extension.
if(MINGW)
    get_filename_component(compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(license_root "${compiler_dir}/../share/licenses")
    set(runtime_packages gcc-libs winpthreads)
endif()
foreach(package IN LISTS runtime_packages)
    if(IS_DIRECTORY "${license_root}/${package}")
        install(DIRECTORY "${license_root}/${package}" DESTINATION oo_solver/licenses)
    endif()
endforeach()
