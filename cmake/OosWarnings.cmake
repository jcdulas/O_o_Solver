# Compiler warnings applied to every O_o_Solver target through oos_enable_warnings().
#
# -Wall -Wextra -Wpedantic -Werror. The code is written for the GCC 12 subset of C++20 even when a newer GCC builds it.

function(oos_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(OOS_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
            # GCC 12 at -O3 reports, inside the standard library, a std::vector assigned a list of one element
            # (`list = {0}`) as a copy out of the bounds of that list (-Warray-bounds), and the bytes of a number
            # inserted at the end of a std::vector<char> as written into a region of size 0 (-Wstringop-overflow):
            # false positives, gone in GCC 13. The two stay warnings there; later compilers keep them errors.
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
                target_compile_options(${target} PRIVATE -Wno-error=array-bounds -Wno-error=stringop-overflow)
            endif()
        endif()
    endif()
endfunction()

# MinGW: link the C++ and GCC runtimes statically so that a built extension or test needs no MinGW DLL.
function(oos_link_static_runtime target)
    if(MINGW)
        target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++)
    endif()
endfunction()
