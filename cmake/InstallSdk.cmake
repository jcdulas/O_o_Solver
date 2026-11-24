include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(sdk_libraries)
foreach(module IN ITEMS base linalg input elements assembly reduction aero output solutions design api)
    set(target oos_${module})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${module})
    target_compile_features(${target} PUBLIC cxx_std_20)
    list(APPEND sdk_libraries ${target})
endforeach()
set(config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/oo-solver")
install(TARGETS ${sdk_libraries} EXPORT SdkTargets ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/oos DESTINATION include PATTERN "ModelFromDeck.hpp" EXCLUDE)
if(OOS_WITH_MODEL)
    install(FILES include/oos/api/ModelFromDeck.hpp DESTINATION include/oos/api)
endif()
install(EXPORT SdkTargets FILE oo-solver-targets.cmake NAMESPACE oos:: DESTINATION "${config_dir}")
configure_package_config_file("cmake/oo-solver-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/oo-solver-config.cmake" INSTALL_DESTINATION "${config_dir}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/oo-solver-config-version.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/oo-solver-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/oo-solver-config-version.cmake" DESTINATION "${config_dir}")
