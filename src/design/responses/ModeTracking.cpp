#include "oos/design/responses/ModeTracking.hpp"

#include <cmath>

namespace oos::design
{

double ModeTracking::modalAssuranceCriterion(std::span<const double> a, std::span<const double> b) noexcept
{
    double ab = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        ab += a[k] * b[k];
        aa += a[k] * a[k];
        bb += b[k] * b[k];
    }
    if (aa == 0.0 || bb == 0.0)
        return 0.0;
    return ab * ab / (aa * bb);
}

ModeTracking::Match ModeTracking::match(std::span<const double> reference, const linalg::DenseMatrix& shapes,
                                        std::size_t count)
{
    const std::size_t columns = count == 0 || count > shapes.columns() ? shapes.columns() : count;
    Match best;
    best.mac = -1.0;
    best.secondMac = -1.0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        const std::span<const double> shape(shapes.data() + j * shapes.leadingDimension(), shapes.rows());
        const double mac = modalAssuranceCriterion(reference, shape);
        // The first of equal values keeps its place: the lower mode wins a tie, in a fixed order.
        if (mac > best.mac)
        {
            best.secondMode = best.mode;
            best.secondMac = best.mac;
            best.mode = j;
            best.mac = mac;
        }
        else if (mac > best.secondMac)
        {
            best.secondMode = j;
            best.secondMac = mac;
        }
    }
    if (best.secondMac < 0.0)
    {
        best.secondMode = best.mode;
        best.secondMac = 0.0;
    }
    best.isUncertain = best.mac < MODE_TRACKING_MIN_MAC || best.mac - best.secondMac < MODE_TRACKING_AMBIGUITY;
    return best;
}

} // namespace oos::design
