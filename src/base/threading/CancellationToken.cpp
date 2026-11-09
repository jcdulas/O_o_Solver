#include "oos/base/threading/CancellationToken.hpp"

#include "oos/base/errors/OperationCancelledError.hpp"

namespace oos::base
{

void CancellationToken::throwIfCancelled() const
{
    if (isCancelled())
        throw OperationCancelledError();
}

} // namespace oos::base
