#include "CheckRunner.hpp"

#include <string>

// Runs the native checks: `oos_native_checks [filter]` runs the checks whose name contains the filter.
int main(int argc, char** argv)
{
    const std::string filter = argc > 1 ? argv[1] : "";
    oos::checks::CheckRunner runner;
    return runner.run(filter) == 0 ? 0 : 1;
}
