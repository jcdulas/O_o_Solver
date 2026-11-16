#include "CheckRunner.hpp"

// The runner itself: a passing expectation records nothing.
CHECK_CASE("runner: a true expectation passes")
{
    runner.expect(true, "a true condition");
    runner.expectNear(1.0 + 1e-15, 1.0, 1e-12, "a close value");
}
